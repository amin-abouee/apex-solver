//! Extended Unified Camera Model (EUCM)
//!
//! An extension of the Unified Camera Model with an additional parameter for
//! improved modeling of wide-angle and fisheye lenses.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! d = √(β(x² + y²) + z²)
//! denom = α·d + (1-α)·z
//! u = fx · (x/denom) + cx
//! v = fy · (y/denom) + cy
//! ```
//!
//! where:
//! - α ∈ [0, 1] is the projection parameter
//! - β > 0 is the distortion parameter
//! - (fx, fy, cx, cy) are standard intrinsics
//!
//! ## Unprojection (2D → 3D)
//!
//! Uses algebraic solution to recover the 3D ray direction.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: α (projection), β (distortion) (6 parameters total)
//!
//! # Use Cases
//!
//! - Wide-angle cameras
//! - Fisheye lenses
//! - More flexible than UCM due to β parameter
//!
//! # References
//!
//! - Khomutenko et al., "An Enhanced Unified Camera Model"

use crate::{CameraModel, CameraModelError, DistortionModel, PinholeParams, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Extended Unified Camera Model with 6 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EucmCamera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl EucmCamera {
    /// Create a new Extended Unified Camera Model (EUCM) camera.
    ///
    /// # Arguments
    ///
    /// * `pinhole` - Pinhole parameters (fx, fy, cx, cy).
    /// * `distortion` - MUST be [`DistortionModel::EUCM`] with `alpha` and `beta`.
    ///
    /// # Returns
    ///
    /// Returns a new `EucmCamera` instance if the distortion model matches.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::InvalidParams`] if `distortion` is not [`DistortionModel::EUCM`].
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

    /// Helper method to extract distortion parameters.
    ///
    /// # Returns
    ///
    /// Returns a tuple `(alpha, beta)` containing the EUCM parameters.
    /// If the distortion model is incorrect (which shouldn't happen for valid instances), returns `(0.0, 0.0)`.
    fn distortion_params(&self) -> (f64, f64) {
        match self.distortion {
            DistortionModel::EUCM { alpha, beta } => (alpha, beta),
            _ => (0.0, 0.0),
        }
    }

    /// Checks the geometric condition for a valid projection.
    ///
    /// # Arguments
    ///
    /// * `z` - The z-coordinate of the point.
    /// * `denom` - The projection denominator `α·d + (1-α)·z`.
    ///
    /// # Returns
    ///
    /// Returns `true` if the point satisfies the projection condition, `false` otherwise.
    fn check_projection_condition(&self, z: f64, denom: f64) -> bool {
        let (alpha, _) = self.distortion_params();
        let mut condition = true;
        if alpha > 0.5 {
            let c = (alpha - 1.0) / (2.0 * alpha - 1.0);
            if z < denom * c {
                condition = false;
            }
        }
        condition
    }

    /// Checks the geometric condition for a valid unprojection.
    ///
    /// # Arguments
    ///
    /// * `r_squared` - The squared radius in normalized image coordinates.
    ///
    /// # Returns
    ///
    /// Returns `true` if the point satisfies the unprojection condition, `false` otherwise.
    fn check_unprojection_condition(&self, r_squared: f64) -> bool {
        let (alpha, beta) = self.distortion_params();
        let mut condition = true;
        if alpha > 0.5 && r_squared > (1.0 / beta * (2.0 * alpha - 1.0)) {
            condition = false;
        }
        condition
    }

    /// Performs linear estimation to initialize distortion parameters from point correspondences.
    ///
    /// This method estimates the `alpha` parameter using a linear least squares approach
    /// given 3D-2D point correspondences. The `beta` parameter is fixed to 1.0.
    /// It assumes the intrinsic parameters (fx, fy, cx, cy) are already set.
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
        if points_2d.ncols() != points_3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        let num_points = points_2d.ncols();
        if num_points < 1 {
            return Err(CameraModelError::InvalidParams(
                "Need at least 1 point for EUCM linear estimation".to_string(),
            ));
        }

        // Set up the linear system to solve for alpha only
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

            b[i * 2] = self.pinhole.fx * x - u_cx * z;
            b[i * 2 + 1] = self.pinhole.fy * y - v_cy * z;
        }

        // Solve the linear system using SVD
        let svd = a.svd(true, true);
        let solution = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol,
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError {
                    operation: "svd_solve".to_string(),
                    details: err_msg.to_string(),
                });
            }
        };

        // Set beta to 1.0 (following reference implementation)
        self.distortion = DistortionModel::EUCM {
            alpha: solution[0],
            beta: 1.0,
        };

        // Validate parameters
        self.validate_params()?;

        Ok(())
    }
}

/// Convert camera to dynamic vector of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, alpha, beta]
impl From<&EucmCamera> for DVector<f64> {
    fn from(camera: &EucmCamera) -> Self {
        let (alpha, beta) = camera.distortion_params();
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            alpha,
            beta,
        ])
    }
}

/// Convert camera to fixed-size array of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, alpha, beta]
impl From<&EucmCamera> for [f64; 6] {
    fn from(camera: &EucmCamera) -> Self {
        let (alpha, beta) = camera.distortion_params();
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            alpha,
            beta,
        ]
    }
}

/// Create camera from slice of intrinsic parameters.
///
/// # Layout
///
/// Expected parameter order: [fx, fy, cx, cy, alpha, beta]
///
/// # Panics
///
/// Panics if the slice has fewer than 6 elements.
impl From<&[f64]> for EucmCamera {
    fn from(params: &[f64]) -> Self {
        assert!(
            params.len() >= 6,
            "EucmCamera requires at least 6 parameters, got {}",
            params.len()
        );
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::EUCM {
                alpha: params[4],
                beta: params[5],
            },
        }
    }
}

/// Create camera from fixed-size array of intrinsic parameters.
///
/// # Layout
///
/// Expected parameter order: [fx, fy, cx, cy, alpha, beta]
impl From<[f64; 6]> for EucmCamera {
    fn from(params: [f64; 6]) -> Self {
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::EUCM {
                alpha: params[4],
                beta: params[5],
            },
        }
    }
}

/// Creates an `EucmCamera` from a parameter slice with validation.
///
/// Unlike `From<&[f64]>`, this constructor validates all parameters
/// and returns a `Result` instead of panicking on invalid input.
///
/// # Errors
///
/// Returns `CameraModelError::InvalidParams` if fewer than 6 parameters are provided.
/// Returns validation errors if focal lengths are non-positive or alpha/beta are out of range.
pub fn try_from_params(params: &[f64]) -> Result<EucmCamera, CameraModelError> {
    if params.len() < 6 {
        return Err(CameraModelError::InvalidParams(format!(
            "EucmCamera requires at least 6 parameters, got {}",
            params.len()
        )));
    }
    let camera = EucmCamera::from(params);
    camera.validate_params()?;
    Ok(camera)
}

impl CameraModel for EucmCamera {
    const INTRINSIC_DIM: usize = 6;
    type IntrinsicJacobian = SMatrix<f64, 2, 6>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// d = √(β(x² + y²) + z²)
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
    /// Returns [`CameraModelError::InvalidParams`] if the geometric projection condition fails or the denominator is too small.
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (alpha, beta) = self.distortion_params();
        let r2 = x * x + y * y;
        let d = (beta * r2 + z * z).sqrt();
        let denom = alpha * d + (1.0 - alpha) * z;

        if denom < crate::GEOMETRIC_PRECISION {
            return Err(CameraModelError::DenominatorTooSmall {
                denom,
                threshold: crate::GEOMETRIC_PRECISION,
            });
        }

        if !self.check_projection_condition(z, denom) {
            return Err(CameraModelError::PointBehindCamera {
                z,
                min_z: crate::GEOMETRIC_PRECISION,
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
    /// Algebraic solution using EUCM inverse equations with α and β parameters.
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
    /// Returns [`CameraModelError::NumericalError`] if a division by zero occurs during calculation.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;

        let (alpha, beta) = self.distortion_params();
        let mx = (u - self.pinhole.cx) / self.pinhole.fx;
        let my = (v - self.pinhole.cy) / self.pinhole.fy;

        let r2 = mx * mx + my * my;
        let beta_r2 = beta * r2;

        let gamma = 1.0 - alpha;
        let gamma_sq = gamma * gamma;

        let discriminant = beta_r2 * gamma_sq + gamma_sq;
        if discriminant < 0.0 || !self.check_unprojection_condition(r2) {
            return Err(CameraModelError::PointOutsideImage { x: u, y: v });
        }

        let sqrt_disc = discriminant.sqrt();
        let denom = beta_r2 + 1.0;

        if denom.abs() < crate::GEOMETRIC_PRECISION {
            return Err(CameraModelError::NumericalError {
                operation: "unprojection".to_string(),
                details: "Division by near-zero in EUCM unprojection".to_string(),
            });
        }

        let mz = (gamma * sqrt_disc) / denom;

        let point3d = Vector3::new(mx, my, mz);
        Ok(point3d.normalize())
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// Computes ∂π/∂p where π is the projection function and p = (x, y, z) is the 3D point.
    ///
    /// # Mathematical Derivation
    ///
    /// For the EUCM camera model, projection is defined as:
    ///
    /// ```text
    /// r² = x² + y²
    /// d = √(β·r² + z²)
    /// denom = α·d + (1-α)·z
    /// u = fx · (x/denom) + cx
    /// v = fy · (y/denom) + cy
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
    /// - u = fx · (x/denom) + cx
    /// - v = fy · (y/denom) + cy
    ///
    /// We apply the quotient rule:
    ///
    /// ```text
    /// ∂(x/denom)/∂x = (denom - x·∂denom/∂x) / denom²
    /// ∂(x/denom)/∂y = -x·∂denom/∂y / denom²
    /// ∂(x/denom)/∂z = -x·∂denom/∂z / denom²
    ///
    /// ∂(y/denom)/∂x = -y·∂denom/∂x / denom²
    /// ∂(y/denom)/∂y = (denom - y·∂denom/∂y) / denom²
    /// ∂(y/denom)/∂z = -y·∂denom/∂z / denom²
    /// ```
    ///
    /// ## Computing Intermediate Derivatives
    ///
    /// First, compute ∂d/∂p where d = √(β·r² + z²):
    ///
    /// ```text
    /// ∂d/∂x = ∂/∂x √(β·(x²+y²) + z²)
    ///       = (1/2) · (β·r² + z²)^(-1/2) · 2β·x
    ///       = β·x / d
    ///
    /// ∂d/∂y = β·y / d
    /// ∂d/∂z = z / d
    /// ```
    ///
    /// Then, compute ∂denom/∂p where denom = α·d + (1-α)·z:
    ///
    /// ```text
    /// ∂denom/∂x = α · ∂d/∂x = α·β·x/d
    /// ∂denom/∂y = α · ∂d/∂y = α·β·y/d
    /// ∂denom/∂z = α · ∂d/∂z + (1-α) = α·z/d + (1-α)
    /// ```
    ///
    /// ## Final Jacobian
    ///
    /// Substituting into the quotient rule expressions:
    ///
    /// ```text
    /// ∂u/∂x = fx · (denom - x·α·β·x/d) / denom²
    /// ∂u/∂y = fx · (-x·α·β·y/d) / denom²
    /// ∂u/∂z = fx · (-x·(α·z/d + 1-α)) / denom²
    ///
    /// ∂v/∂x = fy · (-y·α·β·x/d) / denom²
    /// ∂v/∂y = fy · (denom - y·α·β·y/d) / denom²
    /// ∂v/∂z = fy · (-y·(α·z/d + 1-α)) / denom²
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
    /// - Khomutenko et al., "An Enhanced Unified Camera Model", RAL 2016
    /// - Mei & Rives, "Single View Point Omnidirectional Camera Calibration from Planar Grids", ICRA 2007
    ///
    /// # Numerical Verification
    ///
    /// This analytical Jacobian is verified against numerical differentiation in
    /// `test_jacobian_point_numerical()` with tolerance < 1e-5.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (alpha, beta) = self.distortion_params();
        let r2 = x * x + y * y;
        let d = (beta * r2 + z * z).sqrt();
        let denom = alpha * d + (1.0 - alpha) * z;

        // ∂d/∂x = β·x/d, ∂d/∂y = β·y/d, ∂d/∂z = z/d
        let dd_dx = beta * x / d;
        let dd_dy = beta * y / d;
        let dd_dz = z / d;

        // ∂denom/∂x = α·∂d/∂x
        let ddenom_dx = alpha * dd_dx;
        let ddenom_dy = alpha * dd_dy;
        let ddenom_dz = alpha * dd_dz + (1.0 - alpha);

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
    /// # Mathematical Derivation
    ///
    /// The camera pose transformation converts a world point to camera coordinates:
    ///
    /// ```text
    /// p_cam = T⁻¹ · p_world = R^T · (p_world - t)
    /// ```
    ///
    /// where T = (R, t) is the camera pose (world-to-camera transform).
    ///
    /// ## Perturbation Model (Right Jacobian)
    ///
    /// We perturb the pose in the tangent space of SE(3):
    ///
    /// ```text
    /// T(δξ) = T · exp(δξ^)
    /// ```
    ///
    /// where δξ = (δω, δv) ∈ ℝ⁶ with:
    /// - δω ∈ ℝ³: rotation perturbation (so(3) algebra)
    /// - δv ∈ ℝ³: translation perturbation
    ///
    /// The perturbed camera-frame point becomes:
    ///
    /// ```text
    /// p_cam(δξ) = [T · exp(δξ^)]⁻¹ · p_world
    ///           = exp(-δξ^) · T⁻¹ · p_world
    ///           ≈ (I - δξ^) · p_cam     (first-order approximation)
    /// ```
    ///
    /// ## Jacobian w.r.t. Pose Perturbation
    ///
    /// For small perturbations δξ:
    ///
    /// ```text
    /// p_cam(δξ) ≈ p_cam - [p_cam]× · δω - R^T · δv
    /// ```
    ///
    /// where [p_cam]× is the skew-symmetric matrix of p_cam.
    ///
    /// Taking derivatives:
    ///
    /// ```text
    /// ∂p_cam/∂δω = -[p_cam]×
    /// ∂p_cam/∂δv = -R^T
    /// ```
    ///
    /// Therefore, the Jacobian of p_cam w.r.t. pose perturbation δξ is:
    ///
    /// ```text
    /// J_pose = ∂p_cam/∂δξ = [ -R^T | [p_cam]× ]  (3×6 matrix)
    /// ```
    ///
    /// where:
    /// - First 3 columns correspond to translation perturbation δv
    /// - Last 3 columns correspond to rotation perturbation δω
    ///
    /// ## Chain Rule to Pixel Coordinates
    ///
    /// The full Jacobian chain is:
    ///
    /// ```text
    /// J_pixel_pose = J_pixel_point · J_point_pose
    ///              = (∂u/∂p_cam) · (∂p_cam/∂δξ)
    /// ```
    ///
    /// where J_pixel_point is computed by `jacobian_point()`.
    ///
    /// ## Return Value
    ///
    /// Returns a tuple `(J_pixel_point, J_point_pose)`:
    /// - `J_pixel_point`: 2×3 Jacobian ∂uv/∂p_cam (from jacobian_point)
    /// - `J_point_pose`: 3×6 Jacobian ∂p_cam/∂δξ
    ///
    /// The caller multiplies these to get the full 2×6 Jacobian ∂uv/∂δξ.
    ///
    /// ## SE(3) Conventions
    ///
    /// - **Parameterization**: δξ = [δv_x, δv_y, δv_z, δω_x, δω_y, δω_z]
    /// - **Perturbation**: Right perturbation T(δξ) = T · exp(δξ^)
    /// - **Coordinate frame**: Perturbations are in the camera frame
    ///
    /// ## References
    ///
    /// - Barfoot, "State Estimation for Robotics", Chapter 7 (Lie group optimization)
    /// - Sola et al., "A micro Lie theory for state estimation in robotics", arXiv:1812.01537
    /// - Blanco, "A tutorial on SE(3) transformation parameterizations and on-manifold optimization"
    ///
    /// ## Implementation Notes
    ///
    /// The skew-symmetric matrix [p_cam]× is computed as:
    ///
    /// ```text
    /// [p_cam]× = [  0      -p_z    p_y  ]
    ///            [  p_z     0     -p_x  ]
    ///            [ -p_y    p_x     0   ]
    /// ```
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
    /// # Mathematical Derivation
    ///
    /// The EUCM camera has 6 intrinsic parameters: [fx, fy, cx, cy, α, β]
    ///
    /// ## Projection Equations
    ///
    /// ```text
    /// u = fx · (x/denom) + cx
    /// v = fy · (y/denom) + cy
    /// ```
    ///
    /// where denom = α·d + (1-α)·z and d = √(β·r² + z²).
    ///
    /// ## Jacobian Structure
    ///
    /// We need to compute:
    ///
    /// ```text
    /// J_intrinsics = [ ∂u/∂fx  ∂u/∂fy  ∂u/∂cx  ∂u/∂cy  ∂u/∂α  ∂u/∂β ]
    ///                [ ∂v/∂fx  ∂v/∂fy  ∂v/∂cx  ∂v/∂cy  ∂v/∂α  ∂v/∂β ]
    /// ```
    ///
    /// ## Linear Parameters (fx, fy, cx, cy)
    ///
    /// These appear linearly in the projection equations:
    ///
    /// ```text
    /// ∂u/∂fx = x/denom,   ∂u/∂fy = 0,         ∂u/∂cx = 1,   ∂u/∂cy = 0
    /// ∂v/∂fx = 0,         ∂v/∂fy = y/denom,   ∂v/∂cx = 0,   ∂v/∂cy = 1
    /// ```
    ///
    /// ## Distortion Parameter α
    ///
    /// The parameter α affects denom = α·d + (1-α)·z. Taking derivative:
    ///
    /// ```text
    /// ∂denom/∂α = d - z
    /// ```
    ///
    /// Using the quotient rule for u = fx·(x/denom) + cx:
    ///
    /// ```text
    /// ∂u/∂α = fx · ∂(x/denom)/∂α
    ///       = fx · (-x · ∂denom/∂α) / denom²
    ///       = -fx · x · (d - z) / denom²
    ///
    /// ∂v/∂α = -fy · y · (d - z) / denom²
    /// ```
    ///
    /// ## Distortion Parameter β
    ///
    /// The parameter β affects d = √(β·r² + z²). Taking derivative:
    ///
    /// ```text
    /// ∂d/∂β = ∂/∂β √(β·r² + z²)
    ///       = (1/2) · (β·r² + z²)^(-1/2) · r²
    ///       = r² / (2d)
    /// ```
    ///
    /// Then, using chain rule through denom = α·d + (1-α)·z:
    ///
    /// ```text
    /// ∂denom/∂β = α · ∂d/∂β = α · r² / (2d)
    /// ```
    ///
    /// Finally, using the quotient rule:
    ///
    /// ```text
    /// ∂u/∂β = fx · (-x · ∂denom/∂β) / denom²
    ///       = -fx · x · α · r² / (2d · denom²)
    ///
    /// ∂v/∂β = -fy · y · α · r² / (2d · denom²)
    /// ```
    ///
    /// ## Matrix Form
    ///
    /// The complete Jacobian matrix is:
    ///
    /// ```text
    /// J = [ x/denom    0        1    0    ∂u/∂α    ∂u/∂β ]
    ///     [   0     y/denom    0    1    ∂v/∂α    ∂v/∂β ]
    /// ```
    ///
    /// where:
    /// - ∂u/∂α = -fx · x · (d - z) / denom²
    /// - ∂u/∂β = -fx · x · α · r² / (2d · denom²)
    /// - ∂v/∂α = -fy · y · (d - z) / denom²
    /// - ∂v/∂β = -fy · y · α · r² / (2d · denom²)
    ///
    /// ## References
    ///
    /// - Khomutenko et al., "An Enhanced Unified Camera Model", RAL 2016
    /// - Scaramuzza et al., "A Toolbox for Easily Calibrating Omnidirectional Cameras", IROS 2006
    ///
    /// ## Numerical Verification
    ///
    /// This analytical Jacobian is verified against numerical differentiation in
    /// `test_jacobian_intrinsics_numerical()` with tolerance < 1e-5.
    ///
    /// ## Notes
    ///
    /// The EUCM model parameters have physical interpretation:
    /// - α ∈ [0, 1]: Projection model parameter (α=0 is perspective, α=1 is parabolic)
    /// - β > 0: Mirror parameter controlling field of view
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (alpha, beta) = self.distortion_params();
        let r2 = x * x + y * y;
        let d = (beta * r2 + z * z).sqrt();
        let denom = alpha * d + (1.0 - alpha) * z;

        let x_norm = x / denom;
        let y_norm = y / denom;

        // ∂u/∂fx = x/denom, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = y/denom, ∂v/∂cx = 0, ∂v/∂cy = 1

        // For α and β, need chain rule
        let ddenom_dalpha = d - z;

        let dd_dbeta = r2 / (2.0 * d);
        let ddenom_dbeta = alpha * dd_dbeta;

        let du_dalpha = -self.pinhole.fx * x * ddenom_dalpha / (denom * denom);
        let dv_dalpha = -self.pinhole.fy * y * ddenom_dalpha / (denom * denom);

        let du_dbeta = -self.pinhole.fx * x * ddenom_dbeta / (denom * denom);
        let dv_dbeta = -self.pinhole.fy * y * ddenom_dbeta / (denom * denom);

        SMatrix::<f64, 2, 6>::new(
            x_norm, 0.0, 1.0, 0.0, du_dalpha, du_dbeta, 0.0, y_norm, 0.0, 1.0, dv_dalpha, dv_dbeta,
        )
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - `fx`, `fy` must be positive.
    /// - `fx`, `fy` must be finite.
    /// - `cx`, `cy` must be finite.
    /// - `α` must be in [0, 1].
    /// - `β` must be positive (> 0).
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

        let (alpha, beta) = self.distortion_params();
        if !(alpha.is_finite() && (0.0..=1.0).contains(&alpha)) {
            return Err(CameraModelError::ParameterOutOfRange {
                param: "alpha".to_string(),
                value: alpha,
                min: 0.0,
                max: 1.0,
            });
        }

        if !beta.is_finite() || beta <= 0.0 {
            return Err(CameraModelError::ParameterOutOfRange {
                param: "beta".to_string(),
                value: beta,
                min: 0.0,
                max: f64::INFINITY,
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
    /// The [`DistortionModel`] associated with this camera (typically [`DistortionModel::EUCM`]).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the string identifier for the camera model.
    ///
    /// # Returns
    ///
    /// The string `"eucm"`.
    fn get_model_name(&self) -> &'static str {
        "eucm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2xX, Matrix3xX};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_eucm_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::EUCM {
            alpha: 0.5,
            beta: 1.0,
        };
        let camera = EucmCamera::new(pinhole, distortion)?;

        assert_eq!(camera.pinhole.fx, 300.0);
        assert_eq!(camera.distortion_params(), (0.5, 1.0));
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::EUCM {
            alpha: 0.5,
            beta: 1.0,
        };
        let camera = EucmCamera::new(pinhole, distortion)?;

        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;

        assert!((uv.x - 320.0).abs() < crate::PROJECTION_TEST_TOLERANCE);
        assert!((uv.y - 240.0).abs() < crate::PROJECTION_TEST_TOLERANCE);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::EUCM {
            alpha: 0.6,
            beta: 1.2,
        };
        let camera = EucmCamera::new(pinhole, distortion)?;

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
        let distortion = DistortionModel::EUCM {
            alpha: 0.6,
            beta: 1.2,
        };
        let camera = EucmCamera::new(pinhole, distortion)?;

        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params: DVector<f64> = (&camera).into();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..6 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = EucmCamera::from(params_plus.as_slice());
            let cam_minus = EucmCamera::from(params_minus.as_slice());

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
    fn test_eucm_from_into_traits() -> TestResult {
        let pinhole = PinholeParams::new(400.0, 410.0, 320.0, 240.0)?;
        let distortion = DistortionModel::EUCM {
            alpha: 0.7,
            beta: 1.5,
        };
        let camera = EucmCamera::new(pinhole, distortion)?;

        // Test conversion to DVector
        let params: DVector<f64> = (&camera).into();
        assert_eq!(params.len(), 6);
        assert_eq!(params[0], 400.0);
        assert_eq!(params[1], 410.0);
        assert_eq!(params[2], 320.0);
        assert_eq!(params[3], 240.0);
        assert_eq!(params[4], 0.7);
        assert_eq!(params[5], 1.5);

        // Test conversion to array
        let arr: [f64; 6] = (&camera).into();
        assert_eq!(arr, [400.0, 410.0, 320.0, 240.0, 0.7, 1.5]);

        // Test conversion from slice
        let params_slice = [450.0, 460.0, 330.0, 250.0, 0.8, 1.8];
        let camera2 = EucmCamera::from(&params_slice[..]);
        assert_eq!(camera2.pinhole.fx, 450.0);
        assert_eq!(camera2.pinhole.fy, 460.0);
        assert_eq!(camera2.pinhole.cx, 330.0);
        assert_eq!(camera2.pinhole.cy, 250.0);
        assert_eq!(camera2.distortion_params(), (0.8, 1.8));

        // Test conversion from array
        let camera3 = EucmCamera::from([500.0, 510.0, 340.0, 260.0, 0.9, 2.0]);
        assert_eq!(camera3.pinhole.fx, 500.0);
        assert_eq!(camera3.pinhole.fy, 510.0);
        assert_eq!(camera3.distortion_params(), (0.9, 2.0));

        Ok(())
    }

    #[test]
    fn test_linear_estimation() -> TestResult {
        // Ground truth EUCM camera with beta=1.0 (linear_estimation fixes beta=1.0)
        let gt_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let gt_distortion = DistortionModel::EUCM {
            alpha: 0.5,
            beta: 1.0,
        };
        let gt_camera = EucmCamera::new(gt_pinhole, gt_distortion)?;

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

        // Initial camera with zero alpha and beta=1.0
        let init_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let init_distortion = DistortionModel::EUCM {
            alpha: 0.0,
            beta: 1.0,
        };
        let mut camera = EucmCamera::new(init_pinhole, init_distortion)?;

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
