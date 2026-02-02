//! Kannala-Brandt Fisheye Camera Model
//!
//! A widely-used fisheye camera model with polynomial radial distortion,
//! commonly implemented in OpenCV for fisheye lens calibration.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! r = √(x² + y²)
//! θ = atan2(r, z)
//! θ_d = θ·(1 + k₁·θ² + k₂·θ⁴ + k₃·θ⁶ + k₄·θ⁸)
//! u = fx · θ_d · (x/r) + cx
//! v = fy · θ_d · (y/r) + cy
//! ```
//!
//! Or equivalently: d(θ) = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹
//!
//! ## Unprojection (2D → 3D)
//!
//! Uses Newton-Raphson iteration to solve for θ from θ_d, then recovers
//! the 3D ray direction.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: k₁, k₂, k₃, k₄ (8 parameters total)
//!
//! # Use Cases
//!
//! - Fisheye cameras with up to 180° field of view
//! - Wide-angle surveillance cameras
//! - Automotive and robotics applications
//! - OpenCV fisheye calibration
//!
//! # References
//!
//! - Kannala & Brandt, "A Generic Camera Model and Calibration Method for
//!   Conventional, Wide-Angle, and Fish-Eye Lenses", PAMI 2006

use crate::{skew_symmetric, CameraModel, CameraModelError, DistortionModel, PinholeParams};
use apex_manifolds::se3::SE3;
use apex_manifolds::LieGroup;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Kannala-Brandt fisheye camera model with 8 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KannalaBrandtCamera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl KannalaBrandtCamera {
    /// Create a new Kannala-Brandt fisheye camera.
    ///
    /// # Arguments
    ///
    /// * `pinhole` - Pinhole parameters (fx, fy, cx, cy).
    /// * `distortion` - MUST be [`DistortionModel::KannalaBrandt`] with `k1`, `k2`, `k3`, `k4`.
    ///
    /// # Returns
    ///
    /// Returns a new `KannalaBrandtCamera` instance if the distortion model matches.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::InvalidParams`] if `distortion` is not [`DistortionModel::KannalaBrandt`].
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

    /// Helper method to extract distortion parameters.
    ///
    /// # Returns
    ///
    /// Returns a tuple `(k1, k2, k3, k4)` of distortion coefficients.
    /// If the distortion model is incorrect (which shouldn't happen for valid instances), returns zeros.
    fn distortion_params(&self) -> (f64, f64, f64, f64) {
        match self.distortion {
            DistortionModel::KannalaBrandt { k1, k2, k3, k4 } => (k1, k2, k3, k4),
            _ => (0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Checks the geometric condition for a valid projection.
    ///
    /// # Arguments
    ///
    /// * `z` - The z-coordinate of the point in the camera frame.
    ///
    /// # Returns
    ///
    /// Returns `true` if `z > f64::EPSILON`, `false` otherwise.
    fn check_projection_condition(&self, z: f64) -> bool {
        z > f64::EPSILON
    }

    /// Performs linear estimation to initialize distortion parameters from point correspondences.
    ///
    /// This method estimates the distortion coefficients [k1, k2, k3, k4] using a linear
    /// least squares approach given 3D-2D point correspondences. It assumes the intrinsic
    /// parameters (fx, fy, cx, cy) are already set.
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
        if points_3d.ncols() != points_2d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }
        if points_3d.ncols() < 4 {
            return Err(CameraModelError::InvalidParams(
                "Not enough points for linear estimation (need at least 4)".to_string(),
            ));
        }

        let num_points = points_3d.ncols();
        let mut a_mat = nalgebra::DMatrix::zeros(num_points * 2, 4);
        let mut b_vec = nalgebra::DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let p3d = points_3d.column(i);
            let p2d = points_2d.column(i);

            let x_world = p3d.x;
            let y_world = p3d.y;
            let z_world = p3d.z;

            let u_img = p2d.x;
            let v_img = p2d.y;

            if z_world <= f64::EPSILON {
                continue;
            }

            let r_world = (x_world * x_world + y_world * y_world).sqrt();
            let theta = r_world.atan2(z_world);

            let theta2 = theta * theta;
            let theta3 = theta2 * theta;
            let theta5 = theta3 * theta2;
            let theta7 = theta5 * theta2;
            let theta9 = theta7 * theta2;

            a_mat[(i * 2, 0)] = theta3;
            a_mat[(i * 2, 1)] = theta5;
            a_mat[(i * 2, 2)] = theta7;
            a_mat[(i * 2, 3)] = theta9;

            a_mat[(i * 2 + 1, 0)] = theta3;
            a_mat[(i * 2 + 1, 1)] = theta5;
            a_mat[(i * 2 + 1, 2)] = theta7;
            a_mat[(i * 2 + 1, 3)] = theta9;

            let x_r = if r_world < f64::EPSILON {
                0.0
            } else {
                x_world / r_world
            };
            let y_r = if r_world < f64::EPSILON {
                0.0
            } else {
                y_world / r_world
            };

            if (self.pinhole.fx * x_r).abs() < f64::EPSILON && x_r.abs() > f64::EPSILON {
                return Err(CameraModelError::NumericalError {
                    operation: "linear_estimation".to_string(),
                    details: "fx * x_r is zero in linear estimation".to_string(),
                });
            }
            if (self.pinhole.fy * y_r).abs() < f64::EPSILON && y_r.abs() > f64::EPSILON {
                return Err(CameraModelError::NumericalError {
                    operation: "linear_estimation".to_string(),
                    details: "fy * y_r is zero in linear estimation".to_string(),
                });
            }

            if x_r.abs() > f64::EPSILON {
                b_vec[i * 2] = (u_img - self.pinhole.cx) / (self.pinhole.fx * x_r) - theta;
            } else {
                b_vec[i * 2] = if (u_img - self.pinhole.cx).abs() < f64::EPSILON {
                    -theta
                } else {
                    0.0
                };
            }

            if y_r.abs() > f64::EPSILON {
                b_vec[i * 2 + 1] = (v_img - self.pinhole.cy) / (self.pinhole.fy * y_r) - theta;
            } else {
                b_vec[i * 2 + 1] = if (v_img - self.pinhole.cy).abs() < f64::EPSILON {
                    -theta
                } else {
                    0.0
                };
            }
        }

        let svd = a_mat.svd(true, true);
        let x_coeffs =
            svd.solve(&b_vec, f64::EPSILON)
                .map_err(|e_str| CameraModelError::NumericalError {
                    operation: "svd_solve".to_string(),
                    details: format!("SVD solve failed in linear estimation: {e_str}"),
                })?;

        self.distortion = DistortionModel::KannalaBrandt {
            k1: x_coeffs[0],
            k2: x_coeffs[1],
            k3: x_coeffs[2],
            k4: x_coeffs[3],
        };

        self.validate_params()?;
        Ok(())
    }
}

/// Convert camera to dynamic vector of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, k1, k2, k3, k4]
impl From<&KannalaBrandtCamera> for DVector<f64> {
    fn from(camera: &KannalaBrandtCamera) -> Self {
        let (k1, k2, k3, k4) = camera.distortion_params();
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            k1,
            k2,
            k3,
            k4,
        ])
    }
}

/// Convert camera to fixed-size array of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, k1, k2, k3, k4]
impl From<&KannalaBrandtCamera> for [f64; 8] {
    fn from(camera: &KannalaBrandtCamera) -> Self {
        let (k1, k2, k3, k4) = camera.distortion_params();
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            k1,
            k2,
            k3,
            k4,
        ]
    }
}

/// Create camera from slice of intrinsic parameters.
///
/// # Layout
///
/// Expected parameter order: [fx, fy, cx, cy, k1, k2, k3, k4]
///
/// # Panics
///
/// Panics if the slice has fewer than 8 elements.
impl From<&[f64]> for KannalaBrandtCamera {
    fn from(params: &[f64]) -> Self {
        assert!(
            params.len() >= 8,
            "KannalaBrandtCamera requires at least 8 parameters, got {}",
            params.len()
        );
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::KannalaBrandt {
                k1: params[4],
                k2: params[5],
                k3: params[6],
                k4: params[7],
            },
        }
    }
}

/// Create camera from fixed-size array of intrinsic parameters.
///
/// # Layout
///
/// Expected parameter order: [fx, fy, cx, cy, k1, k2, k3, k4]
impl From<[f64; 8]> for KannalaBrandtCamera {
    fn from(params: [f64; 8]) -> Self {
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::KannalaBrandt {
                k1: params[4],
                k2: params[5],
                k3: params[6],
                k4: params[7],
            },
        }
    }
}

impl CameraModel for KannalaBrandtCamera {
    const INTRINSIC_DIM: usize = 8;
    type IntrinsicJacobian = SMatrix<f64, 2, 8>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// r = √(x² + y²)
    /// θ = atan2(r, z)
    /// θ_d = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹
    /// u = fx · θ_d · (x/r) + cx
    /// v = fy · θ_d · (y/r) + cy
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
    /// Returns [`CameraModelError::InvalidParams`] if point is behind camera (z <= EPSILON).
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        // Check if point is valid for projection (in front of camera)
        if !self.check_projection_condition(z) {
            return Err(CameraModelError::PointBehindCamera {
                z,
                min_z: f64::EPSILON,
            });
        }

        let (k1, k2, k3, k4) = self.distortion_params();
        let r2 = x * x + y * y;
        let r = r2.sqrt();
        let theta = r.atan2(z);

        // Polynomial distortion: d(θ) = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

        if r < crate::GEOMETRIC_PRECISION {
            // Point near optical axis: x/r and y/r are unstable.
            // Limit approaches (fx * (theta_d/r) * x + cx)
            // theta ~ r/z (for small theta), theta_d ~ theta (for small theta)
            // theta_d/r ~ 1/z.
            // u = fx * x/z + cx, v = fy * y/z + cy.
            // Effectively pinhole close to center.
            let inv_z = 1.0 / z;
            return Ok(Vector2::new(
                self.pinhole.fx * x * inv_z + self.pinhole.cx,
                self.pinhole.fy * y * inv_z + self.pinhole.cy,
            ));
        }

        let inv_r = 1.0 / r;
        Ok(Vector2::new(
            self.pinhole.fx * theta_d * x * inv_r + self.pinhole.cx,
            self.pinhole.fy * theta_d * y * inv_r + self.pinhole.cy,
        ))
    }

    /// Unprojects a 2D image point to a 3D ray.
    ///
    /// # Algorithm
    ///
    /// Newton-Raphson iteration to solve for θ from θ_d:
    /// - f(θ) = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹ - θ_d = 0
    /// - f'(θ) = 1 + 3k₁·θ² + 5k₂·θ⁴ + 7k₃·θ⁶ + 9k₄·θ⁸
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
    /// Returns [`CameraModelError::NumericalError`] if Newton-Raphson fails to converge or derivative is too small.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;

        let (k1, k2, k3, k4) = self.distortion_params();
        let mx = (u - self.pinhole.cx) / self.pinhole.fx;
        let my = (v - self.pinhole.cy) / self.pinhole.fy;

        let mut ru = (mx * mx + my * my).sqrt();

        // Clamp ru to avoid instability if extremely large (from C++ impl: min(ru, PI/2))
        ru = ru.min(std::f64::consts::PI / 2.0);

        if ru < crate::GEOMETRIC_PRECISION {
            return Ok(Vector3::new(0.0, 0.0, 1.0));
        }

        // Newton-Raphson
        let mut theta = ru; // Initial guess
        const MAX_ITER: usize = 10;
        const CONVERGENCE_THRESHOLD: f64 = crate::CONVERGENCE_THRESHOLD;

        for _ in 0..MAX_ITER {
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta4 * theta4;

            let k1_theta2 = k1 * theta2;
            let k2_theta4 = k2 * theta4;
            let k3_theta6 = k3 * theta6;
            let k4_theta8 = k4 * theta8;

            // f(θ)
            let f = theta * (1.0 + k1_theta2 + k2_theta4 + k3_theta6 + k4_theta8) - ru;

            // f'(θ)
            let f_prime =
                1.0 + 3.0 * k1_theta2 + 5.0 * k2_theta4 + 7.0 * k3_theta6 + 9.0 * k4_theta8;

            if f_prime.abs() < f64::EPSILON {
                return Err(CameraModelError::NumericalError {
                    operation: "unprojection".to_string(),
                    details: "Derivative too small in KB unprojection".to_string(),
                });
            }

            let delta = f / f_prime;
            theta -= delta;

            if delta.abs() < CONVERGENCE_THRESHOLD {
                break;
            }
        }

        // Convert θ to 3D ray
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        // Direction in xy plane
        // if ru is small we returned already.
        // x = mx * sin(theta) / ru
        // y = my * sin(theta) / ru
        let scale = sin_theta / ru;
        let x = mx * scale;
        let y = my * scale;
        let z = cos_theta;

        Ok(Vector3::new(x, y, z).normalize())
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Validity Conditions
    ///
    /// - Always returns true (KB model has wide acceptance range)
    ///
    /// # Arguments
    ///
    /// * `_p_cam` - 3D point in camera coordinate frame (unused in this implementation).
    ///
    /// # Returns
    ///
    /// Always returns `true` for the Kannala-Brandt model.
    fn is_valid_point(&self, _p_cam: &Vector3<f64>) -> bool {
        true
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// Computes ∂π/∂p where π is the projection function and p = (x, y, z) is the 3D point.
    ///
    /// # Mathematical Derivation
    ///
    /// ## Kannala-Brandt Projection Model Recap
    ///
    /// ```text
    /// r = √(x² + y²)                              // Radial distance from optical axis
    /// θ = atan2(r, z)                             // Angle from optical axis
    /// θ_d = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹    // Distorted angle (polynomial)
    /// u = fx · θ_d · (x/r) + cx                  // Pixel u-coordinate
    /// v = fy · θ_d · (y/r) + cy                  // Pixel v-coordinate
    /// ```
    ///
    /// ## Step 1: Derivatives of intermediate quantities
    ///
    /// ### Radial distance r:
    /// ```text
    /// ∂r/∂x = x/r
    /// ∂r/∂y = y/r
    /// ∂r/∂z = 0
    /// ```
    ///
    /// ### Angle θ = atan2(r, z):
    /// Using the atan2 derivative formula:
    /// ```text
    /// ∂θ/∂x = ∂(atan2(r,z))/∂x = (z/r²+z²) · ∂r/∂x = z·x / (r·(r²+z²))
    /// ∂θ/∂y = z·y / (r·(r²+z²))
    /// ∂θ/∂z = -r / (r²+z²)
    /// ```
    ///
    /// ### Distorted angle θ_d:
    /// The polynomial distortion is θ_d = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹
    ///
    /// Derivative w.r.t. θ:
    /// ```text
    /// ∂θ_d/∂θ = 1 + 3k₁·θ² + 5k₂·θ⁴ + 7k₃·θ⁶ + 9k₄·θ⁸
    /// ```
    ///
    /// By chain rule:
    /// ```text
    /// ∂θ_d/∂x = (∂θ_d/∂θ) · (∂θ/∂x)
    /// ∂θ_d/∂y = (∂θ_d/∂θ) · (∂θ/∂y)
    /// ∂θ_d/∂z = (∂θ_d/∂θ) · (∂θ/∂z)
    /// ```
    ///
    /// ## Step 2: Derivatives of pixel coordinates (quotient + product rule)
    ///
    /// For u = fx · θ_d · (x/r) + cx:
    /// ```text
    /// ∂u/∂x = fx · [∂θ_d/∂x · (x/r) + θ_d · ∂(x/r)/∂x]
    ///       = fx · [∂θ_d/∂x · (x/r) + θ_d · (1/r - x²/r³)]
    ///
    /// ∂u/∂y = fx · [∂θ_d/∂y · (x/r) + θ_d · (-x·y/r³)]
    ///
    /// ∂u/∂z = fx · [∂θ_d/∂z · (x/r)]
    /// ```
    ///
    /// Similarly for v = fy · θ_d · (y/r) + cy:
    /// ```text
    /// ∂v/∂x = fy · [∂θ_d/∂x · (y/r) + θ_d · (-x·y/r³)]
    ///
    /// ∂v/∂y = fy · [∂θ_d/∂y · (y/r) + θ_d · (1/r - y²/r³)]
    ///
    /// ∂v/∂z = fy · [∂θ_d/∂z · (y/r)]
    /// ```
    ///
    /// ## Step 3: Special case - Near optical axis (r → 0)
    ///
    /// When r is very small, the point is near the optical axis. In this limit:
    /// - x/r, y/r become undefined
    /// - θ → 0, so θ_d → 0
    /// - The projection becomes approximately linear
    ///
    /// For numerical stability, use simplified Jacobian:
    /// ```text
    /// ∂u/∂x ≈ fx · (∂θ_d/∂θ) / z
    /// ∂v/∂y ≈ fy · (∂θ_d/∂θ) / z
    /// (all other terms ≈ 0)
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
    /// Returns the 2×3 Jacobian matrix.
    ///
    /// # References
    ///
    /// - Kannala & Brandt, "A Generic Camera Model and Calibration Method for Conventional,
    ///   Wide-Angle, and Fish-Eye Lenses", IEEE PAMI 2006
    /// - Verified against numerical differentiation in `test_jacobian_point_numerical()`
    ///
    /// # Implementation Note
    ///
    /// The implementation handles the optical axis singularity (r → 0) using a threshold check
    /// and falls back to a simplified Jacobian for numerical stability.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (k1, k2, k3, k4) = self.distortion_params();
        let r = (x * x + y * y).sqrt();
        let theta = r.atan2(z);

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

        // ∂θ_d/∂θ = 1 + 3k₁·θ² + 5k₂·θ⁴ + 7k₃·θ⁶ + 9k₄·θ⁸
        let dtheta_d_dtheta = 1.0
            + 3.0 * k1 * theta2
            + 5.0 * k2 * theta2 * theta2
            + 7.0 * k3 * theta2 * theta2 * theta2
            + 9.0 * k4 * theta2 * theta2 * theta2 * theta2;

        if r < crate::GEOMETRIC_PRECISION {
            // Near optical axis, use simplified Jacobian
            return SMatrix::<f64, 2, 3>::new(
                self.pinhole.fx * dtheta_d_dtheta / z,
                0.0,
                0.0,
                0.0,
                self.pinhole.fy * dtheta_d_dtheta / z,
                0.0,
            );
        }

        let inv_r = 1.0 / r;
        let r2 = r * r;
        let r_z2 = r2 + z * z;

        // ∂θ/∂x = z·x / (r·(r² + z²))
        // ∂θ/∂y = z·y / (r·(r² + z²))
        // ∂θ/∂z = -r / (r² + z²)
        let dtheta_dx = z * x / (r * r_z2);
        let dtheta_dy = z * y / (r * r_z2);
        let dtheta_dz = -r / r_z2;

        // ∂r/∂x = x/r, ∂r/∂y = y/r, ∂r/∂z = 0

        // Chain rule for u = fx · θ_d · (x/r) + cx
        let inv_r2 = inv_r * inv_r;

        let du_dx = self.pinhole.fx
            * (dtheta_d_dtheta * dtheta_dx * x * inv_r
                + theta_d * (inv_r - x * x * inv_r2 * inv_r));
        let du_dy = self.pinhole.fx
            * (dtheta_d_dtheta * dtheta_dy * x * inv_r - theta_d * x * y * inv_r2 * inv_r);
        let du_dz = self.pinhole.fx * dtheta_d_dtheta * dtheta_dz * x * inv_r;

        let dv_dx = self.pinhole.fy
            * (dtheta_d_dtheta * dtheta_dx * y * inv_r - theta_d * x * y * inv_r2 * inv_r);
        let dv_dy = self.pinhole.fy
            * (dtheta_d_dtheta * dtheta_dy * y * inv_r
                + theta_d * (inv_r - y * y * inv_r2 * inv_r));
        let dv_dz = self.pinhole.fy * dtheta_d_dtheta * dtheta_dz * y * inv_r;

        SMatrix::<f64, 2, 3>::new(du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz)
    }

    /// Jacobian of projection w.r.t. camera pose (SE3).
    ///
    /// Computes ∂π/∂ξ where π is the projection function and ξ ∈ se(3) is the camera pose.
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
    /// See that method's documentation for the complete derivation of Kannala-Brandt
    /// projection derivatives.
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

    /// Jacobian of projection w.r.t. intrinsic parameters (2×8).
    ///
    /// Computes ∂π/∂K where K = [fx, fy, cx, cy, k₁, k₂, k₃, k₄] are the intrinsic parameters.
    ///
    /// # Mathematical Derivation
    ///
    /// The intrinsic parameters consist of:
    /// 1. **Linear parameters**: fx, fy, cx, cy (pinhole projection)
    /// 2. **Distortion parameters**: k₁, k₂, k₃, k₄ (Kannala-Brandt polynomial coefficients)
    ///
    /// ## Kannala-Brandt Projection Model Recap
    ///
    /// ```text
    /// r = √(x² + y²)
    /// θ = atan2(r, z)
    /// θ_d = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹
    /// u = fx · θ_d · (x/r) + cx
    /// v = fy · θ_d · (y/r) + cy
    /// ```
    ///
    /// ## Part 1: Linear Parameters (fx, fy, cx, cy)
    ///
    /// These have direct, simple derivatives since they appear linearly in the projection:
    ///
    /// ### Focal lengths (fx, fy):
    /// ```text
    /// ∂u/∂fx = θ_d · (x/r)    (coefficient of fx in u)
    /// ∂u/∂fy = 0              (fy doesn't affect u)
    /// ∂v/∂fx = 0              (fx doesn't affect v)
    /// ∂v/∂fy = θ_d · (y/r)    (coefficient of fy in v)
    /// ```
    ///
    /// ### Principal point (cx, cy):
    /// ```text
    /// ∂u/∂cx = 1              (cx adds directly to u)
    /// ∂u/∂cy = 0              (cy doesn't affect u)
    /// ∂v/∂cx = 0              (cx doesn't affect v)
    /// ∂v/∂cy = 1              (cy adds directly to v)
    /// ```
    ///
    /// ## Part 2: Distortion Parameters (k₁, k₂, k₃, k₄)
    ///
    /// The distortion affects θ_d through the polynomial expansion.
    ///
    /// ### Derivatives of θ_d:
    /// ```text
    /// θ_d = θ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹
    ///
    /// ∂θ_d/∂k₁ = θ³
    /// ∂θ_d/∂k₂ = θ⁵
    /// ∂θ_d/∂k₃ = θ⁷
    /// ∂θ_d/∂k₄ = θ⁹
    /// ```
    ///
    /// ### Chain rule to pixel coordinates:
    ///
    /// For u = fx · θ_d · (x/r) + cx:
    /// ```text
    /// ∂u/∂k₁ = fx · (∂θ_d/∂k₁) · (x/r) = fx · θ³ · (x/r)
    /// ∂u/∂k₂ = fx · θ⁵ · (x/r)
    /// ∂u/∂k₃ = fx · θ⁷ · (x/r)
    /// ∂u/∂k₄ = fx · θ⁹ · (x/r)
    /// ```
    ///
    /// Similarly for v = fy · θ_d · (y/r) + cy:
    /// ```text
    /// ∂v/∂k₁ = fy · θ³ · (y/r)
    /// ∂v/∂k₂ = fy · θ⁵ · (y/r)
    /// ∂v/∂k₃ = fy · θ⁷ · (y/r)
    /// ∂v/∂k₄ = fy · θ⁹ · (y/r)
    /// ```
    ///
    /// ## Final Jacobian Matrix (2×8)
    ///
    /// ```text
    /// J = [ ∂u/∂fx  ∂u/∂fy  ∂u/∂cx  ∂u/∂cy  ∂u/∂k₁  ∂u/∂k₂  ∂u/∂k₃  ∂u/∂k₄ ]
    ///     [ ∂v/∂fx  ∂v/∂fy  ∂v/∂cx  ∂v/∂cy  ∂v/∂k₁  ∂v/∂k₂  ∂v/∂k₃  ∂v/∂k₄ ]
    /// ```
    ///
    /// Expanded:
    /// ```text
    /// J = [ θ_d·x/r    0       1    0    fx·θ³·x/r  fx·θ⁵·x/r  fx·θ⁷·x/r  fx·θ⁹·x/r ]
    ///     [    0    θ_d·y/r    0    1    fy·θ³·y/r  fy·θ⁵·y/r  fy·θ⁷·y/r  fy·θ⁹·y/r ]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame.
    ///
    /// # Returns
    ///
    /// Returns the 2×8 intrinsic Jacobian matrix.
    ///
    /// # References
    ///
    /// - Kannala & Brandt, "A Generic Camera Model and Calibration Method for Conventional,
    ///   Wide-Angle, and Fish-Eye Lenses", IEEE PAMI 2006
    /// - Verified against numerical differentiation in `test_jacobian_intrinsics_numerical()`
    ///
    /// # Implementation Note
    ///
    /// For numerical stability, when r (radial distance) is very small (near optical axis),
    /// the Jacobian is set to zero as the projection becomes degenerate in this region.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (k1, k2, k3, k4) = self.distortion_params();
        let r = (x * x + y * y).sqrt();
        let theta = r.atan2(z);

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

        if r < crate::GEOMETRIC_PRECISION {
            return SMatrix::<f64, 2, 8>::zeros();
        }

        let inv_r = 1.0 / r;
        let x_theta_d_r = x * theta_d * inv_r;
        let y_theta_d_r = y * theta_d * inv_r;

        // ∂u/∂fx = θ_d·x/r, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = θ_d·y/r, ∂v/∂cx = 0, ∂v/∂cy = 1

        // ∂u/∂k₁ = fx·θ³·x/r, ∂u/∂k₂ = fx·θ⁵·x/r, etc.
        let du_dk1 = self.pinhole.fx * theta3 * x * inv_r;
        let du_dk2 = self.pinhole.fx * theta5 * x * inv_r;
        let du_dk3 = self.pinhole.fx * theta7 * x * inv_r;
        let du_dk4 = self.pinhole.fx * theta9 * x * inv_r;

        let dv_dk1 = self.pinhole.fy * theta3 * y * inv_r;
        let dv_dk2 = self.pinhole.fy * theta5 * y * inv_r;
        let dv_dk3 = self.pinhole.fy * theta7 * y * inv_r;
        let dv_dk4 = self.pinhole.fy * theta9 * y * inv_r;

        SMatrix::<f64, 2, 8>::from_row_slice(&[
            x_theta_d_r,
            0.0,
            1.0,
            0.0,
            du_dk1,
            du_dk2,
            du_dk3,
            du_dk4,
            0.0,
            y_theta_d_r,
            0.0,
            1.0,
            dv_dk1,
            dv_dk2,
            dv_dk3,
            dv_dk4,
        ])
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - `fx`, `fy` must be positive.
    /// - `fx`, `fy` must be finite.
    /// - `cx`, `cy` must be finite.
    /// - `k1`..`k4` must be finite.
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

        let (k1, k2, k3, k4) = self.distortion_params();
        if !k1.is_finite() {
            return Err(CameraModelError::DistortionNotFinite {
                name: "k1".to_string(),
                value: k1,
            });
        }
        if !k2.is_finite() {
            return Err(CameraModelError::DistortionNotFinite {
                name: "k2".to_string(),
                value: k2,
            });
        }
        if !k3.is_finite() {
            return Err(CameraModelError::DistortionNotFinite {
                name: "k3".to_string(),
                value: k3,
            });
        }
        if !k4.is_finite() {
            return Err(CameraModelError::DistortionNotFinite {
                name: "k4".to_string(),
                value: k4,
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
    /// The [`DistortionModel`] associated with this camera (typically [`DistortionModel::KannalaBrandt`]).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the string identifier for the camera model.
    ///
    /// # Returns
    ///
    /// The string `"kannala_brandt"`.
    fn get_model_name(&self) -> &'static str {
        "kannala_brandt"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_kb_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::KannalaBrandt {
            k1: 0.1,
            k2: 0.01,
            k3: 0.001,
            k4: 0.0001,
        };
        let camera = KannalaBrandtCamera::new(pinhole, distortion)?;
        assert_eq!(camera.pinhole.fx, 300.0);
        let (k1, _, _, _) = camera.distortion_params();
        assert_eq!(k1, 0.1);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::KannalaBrandt {
            k1: 0.1,
            k2: 0.01,
            k3: 0.001,
            k4: 0.0001,
        };
        let camera = KannalaBrandtCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;

        assert!((uv.x - 320.0).abs() < 1e-6);
        assert!((uv.y - 240.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::KannalaBrandt {
            k1: 0.1,
            k2: 0.01,
            k3: 0.001,
            k4: 0.0001,
        };
        let camera = KannalaBrandtCamera::new(pinhole, distortion)?;
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
        let distortion = DistortionModel::KannalaBrandt {
            k1: 0.1,
            k2: 0.01,
            k3: 0.001,
            k4: 0.0001,
        };
        let camera = KannalaBrandtCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params: DVector<f64> = (&camera).into();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..8 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = KannalaBrandtCamera::from(params_plus.as_slice());
            let cam_minus = KannalaBrandtCamera::from(params_minus.as_slice());

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
    fn test_kb_from_into_traits() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::KannalaBrandt {
            k1: 0.1,
            k2: 0.01,
            k3: 0.001,
            k4: 0.0001,
        };
        let camera = KannalaBrandtCamera::new(pinhole, distortion)?;

        // Test conversion to DVector
        let params: DVector<f64> = (&camera).into();
        assert_eq!(params.len(), 8);
        assert_eq!(params[0], 300.0);
        assert_eq!(params[1], 300.0);
        assert_eq!(params[2], 320.0);
        assert_eq!(params[3], 240.0);
        assert_eq!(params[4], 0.1);
        assert_eq!(params[5], 0.01);
        assert_eq!(params[6], 0.001);
        assert_eq!(params[7], 0.0001);

        // Test conversion to array
        let arr: [f64; 8] = (&camera).into();
        assert_eq!(arr, [300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.0001]);

        // Test conversion from slice
        let params_slice = [350.0, 350.0, 330.0, 250.0, 0.2, 0.02, 0.002, 0.0002];
        let camera2 = KannalaBrandtCamera::from(&params_slice[..]);
        assert_eq!(camera2.pinhole.fx, 350.0);
        assert_eq!(camera2.pinhole.fy, 350.0);
        assert_eq!(camera2.pinhole.cx, 330.0);
        assert_eq!(camera2.pinhole.cy, 250.0);
        let (k1, k2, k3, k4) = camera2.distortion_params();
        assert_eq!(k1, 0.2);
        assert_eq!(k2, 0.02);
        assert_eq!(k3, 0.002);
        assert_eq!(k4, 0.0002);

        // Test conversion from array
        let camera3 =
            KannalaBrandtCamera::from([400.0, 400.0, 340.0, 260.0, 0.3, 0.03, 0.003, 0.0003]);
        assert_eq!(camera3.pinhole.fx, 400.0);
        assert_eq!(camera3.pinhole.fy, 400.0);
        let (k1, k2, k3, k4) = camera3.distortion_params();
        assert_eq!(k1, 0.3);
        assert_eq!(k2, 0.03);
        assert_eq!(k3, 0.003);
        assert_eq!(k4, 0.0003);

        Ok(())
    }
}
