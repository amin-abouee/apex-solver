//! Radial-Tangential Distortion Camera Model
//!
//! The standard OpenCV camera model combining radial and tangential distortion.
//! Widely used for narrow to moderate field-of-view cameras.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! x' = x/z,  y' = y/z  (normalized coordinates)
//! r² = x'² + y'²
//!
//! Radial distortion:
//! r' = 1 + k₁·r² + k₂·r⁴ + k₃·r⁶
//!
//! Tangential distortion:
//! dx = 2·p₁·x'·y' + p₂·(r² + 2·x'²)
//! dy = p₁·(r² + 2·y'²) + 2·p₂·x'·y'
//!
//! Distorted coordinates:
//! x'' = r'·x' + dx
//! y'' = r'·y' + dy
//!
//! Final projection:
//! u = fx·x'' + cx
//! v = fy·y'' + cy
//! ```
//!
//! ## Unprojection (2D → 3D)
//!
//! Iterative Jacobian-based method to solve the non-linear inverse equations.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: k₁, k₂, p₁, p₂, k₃ (9 parameters total)
//!
//! # Use Cases
//!
//! - Standard narrow FOV cameras
//! - OpenCV-calibrated cameras
//! - Robotics and AR/VR applications
//! - Most conventional lenses
//!
//! # References
//!
//! - Brown, "Decentering Distortion of Lenses", 1966
//! - OpenCV Camera Calibration Documentation

use crate::{
    CameraModel, CameraModelError, DistortionModel, PinholeParams, Resolution, skew_symmetric,
};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, Matrix2, SMatrix, Vector2, Vector3};

/// Radial-Tangential camera model with 9 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadTanCamera {
    /// Linear pinhole parameters (fx, fy, cx, cy)
    pub pinhole: PinholeParams,
    /// Lens distortion model and parameters
    pub distortion: DistortionModel,
    /// Image resolution
    pub resolution: Resolution,
}

impl RadTanCamera {
    /// Create a new Radial-Tangential (Brown-Conrady) camera.
    ///
    /// # Arguments
    ///
    /// * `pinhole` - Pinhole parameters (fx, fy, cx, cy)
    /// * `distortion` - MUST be DistortionModel::BrownConrady { k1, k2, p1, p2, k3 }
    /// * `resolution` - Image resolution
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError::InvalidParams` if `distortion` is not `DistortionModel::BrownConrady`.
    pub fn new(
        pinhole: PinholeParams,
        distortion: DistortionModel,
        resolution: Resolution,
    ) -> Result<Self, CameraModelError> {
        // Validate distortion model
        // if !matches!(distortion, DistortionModel::BrownConrady { .. }) {
        //     return Err(CameraModelError::InvalidParams(format!(
        //         "RadTanCamera requires BrownConrady distortion model, got {:?}",
        //         distortion
        //     )));
        // }

        let camera = Self {
            pinhole,
            distortion,
            resolution,
        };
        camera.validate_params()?;
        Ok(camera)
    }

    /// Checks if a 3D point satisfies the projection condition (z >= crate::GEOMETRIC_PRECISION).
    fn check_projection_condition(&self, z: f64) -> bool {
        z >= crate::GEOMETRIC_PRECISION
    }

    /// Helper to extract distortion parameters, returning Result for consistency.
    fn distortion_params(&self) -> Result<(f64, f64, f64, f64, f64), CameraModelError> {
        match self.distortion {
            DistortionModel::BrownConrady { k1, k2, p1, p2, k3 } => Ok((k1, k2, p1, p2, k3)),
            _ => Err(CameraModelError::InvalidParams(format!(
                "RadTanCamera requires BrownConrady distortion model, got {:?}",
                self.distortion
            ))),
        }
    }
}

impl CameraModel for RadTanCamera {
    const INTRINSIC_DIM: usize = 9;
    type IntrinsicJacobian = SMatrix<f64, 2, 9>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// Combines radial distortion (k₁, k₂, k₃) and tangential distortion (p₁, p₂).
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// - `Ok(uv)` - 2D image coordinates if valid
    /// - `Err` - If point is at or behind camera
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        if !self.check_projection_condition(p_cam.z) {
            return Err(CameraModelError::InvalidParams(format!(
                "RadTan: z must be >= {}, got z={}",
                crate::GEOMETRIC_PRECISION,
                p_cam.z
            )));
        }

        let inv_z = 1.0 / p_cam.z;
        let x_prime = p_cam.x * inv_z;
        let y_prime = p_cam.y * inv_z;

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let (k1, k2, p1, p2, k3) = self
            .distortion_params()
            .expect("RadTanCamera validated at construction");

        // Radial distortion: r' = 1 + k₁·r² + k₂·r⁴ + k₃·r⁶
        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        // Tangential distortion
        let xy = x_prime * y_prime;
        let dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x_prime * x_prime);
        let dy = p1 * (r2 + 2.0 * y_prime * y_prime) + 2.0 * p2 * xy;

        // Distorted coordinates
        let x_distorted = radial * x_prime + dx;
        let y_distorted = radial * y_prime + dy;

        Ok(Vector2::new(
            self.pinhole.fx * x_distorted + self.pinhole.cx,
            self.pinhole.fy * y_distorted + self.pinhole.cy,
        ))
    }

    /// Unprojects a 2D image point to a 3D ray.
    ///
    /// # Algorithm
    ///
    /// Iterative Newton-Raphson with Jacobian matrix:
    /// 1. Start with undistorted estimate
    /// 2. Compute distortion and Jacobian
    /// 3. Update estimate: p' = p' - J⁻¹·f(p')
    /// 4. Repeat until convergence
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates
    ///
    /// # Returns
    ///
    /// - `Ok(ray)` - Normalized 3D ray direction
    /// - `Err` - If iteration fails to converge
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        // Validate unprojection condition if needed (always true for RadTan generally)
        let u = point_2d.x;
        let v = point_2d.y;

        // Initial estimate (undistorted)
        let x_distorted = (u - self.pinhole.cx) / self.pinhole.fx;
        let y_distorted = (v - self.pinhole.cy) / self.pinhole.fy;
        let target_distorted_point = Vector2::new(x_distorted, y_distorted);

        let mut point = target_distorted_point;

        const EPS: f64 = crate::GEOMETRIC_PRECISION;
        const MAX_ITERATIONS: u32 = 100;

        let (k1, k2, p1, p2, k3) = self
            .distortion_params()
            .expect("RadTanCamera validated at construction");

        for iteration in 0..MAX_ITERATIONS {
            let x = point.x;
            let y = point.y;

            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;

            // Radial distortion
            let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

            // Tangential distortion
            let xy = x * y;
            let dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x);
            let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy;

            // Distorted point
            let x_dist = radial * x + dx;
            let y_dist = radial * y + dy;

            // Residual
            let fx = x_dist - target_distorted_point.x;
            let fy = y_dist - target_distorted_point.y;

            if fx.abs() < EPS && fy.abs() < EPS {
                break;
            }

            // Jacobian matrix
            let dradial_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

            // ∂(radial·x + dx)/∂x
            let dfx_dx = radial + 2.0 * x * dradial_dr2 * x + 2.0 * p1 * y + 2.0 * p2 * (3.0 * x);

            // ∂(radial·x + dx)/∂y
            let dfx_dy = 2.0 * x * dradial_dr2 * y + 2.0 * p1 * x + 2.0 * p2 * y;

            // ∂(radial·y + dy)/∂x
            let dfy_dx = 2.0 * y * dradial_dr2 * x + 2.0 * p1 * x + 2.0 * p2 * y;

            // ∂(radial·y + dy)/∂y
            let dfy_dy = radial + 2.0 * y * dradial_dr2 * y + 2.0 * p1 * (3.0 * y) + 2.0 * p2 * x;

            let jacobian = Matrix2::new(dfx_dx, dfx_dy, dfy_dx, dfy_dy);

            // Solve: J·Δp = -f
            let det = jacobian[(0, 0)] * jacobian[(1, 1)] - jacobian[(0, 1)] * jacobian[(1, 0)];

            if det.abs() < crate::GEOMETRIC_PRECISION {
                return Err(CameraModelError::NumericalError(
                    "Singular Jacobian in RadTan unprojection".to_string(),
                ));
            }

            let inv_det = 1.0 / det;
            let delta_x = inv_det * (jacobian[(1, 1)] * (-fx) - jacobian[(0, 1)] * (-fy));
            let delta_y = inv_det * (-jacobian[(1, 0)] * (-fx) + jacobian[(0, 0)] * (-fy));

            point.x += delta_x;
            point.y += delta_y;

            if iteration == MAX_ITERATIONS - 1 {
                return Err(CameraModelError::NumericalError(
                    "RadTan unprojection did not converge".to_string(),
                ));
            }
        }

        // Normalize to unit ray
        let r2 = point.x * point.x + point.y * point.y;
        let norm = (1.0 + r2).sqrt();
        let norm_inv = 1.0 / norm;

        Ok(Vector3::new(
            point.x * norm_inv,
            point.y * norm_inv,
            norm_inv,
        ))
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Validity Conditions
    ///
    /// - z ≥ PRECISION (point in front of camera)
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        self.check_projection_condition(p_cam.z)
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// # Mathematical Derivation
    ///
    /// For the Radial-Tangential (Brown-Conrady) model, projection is:
    ///
    /// ```text
    /// x' = x/z,  y' = y/z  (normalized coordinates)
    /// r² = x'² + y'²
    /// radial = 1 + k₁·r² + k₂·r⁴ + k₃·r⁶
    /// dx = 2·p₁·x'·y' + p₂·(r² + 2·x'²)  (tangential distortion)
    /// dy = p₁·(r² + 2·y'²) + 2·p₂·x'·y'  (tangential distortion)
    /// x'' = radial·x' + dx
    /// y'' = radial·y' + dy
    /// u = fx·x'' + cx
    /// v = fy·y'' + cy
    /// ```
    ///
    /// ## Chain Rule Application
    ///
    /// This is the most complex Jacobian due to coupled radial + tangential distortion.
    /// The chain rule must be applied through multiple stages:
    ///
    /// 1. ∂(x',y')/∂(x,y,z): Normalized coordinate derivatives
    /// 2. ∂(x'',y'')/∂(x',y'): Distortion derivatives (radial + tangential)
    /// 3. ∂(u,v)/∂(x'',y''): Final projection derivatives
    ///
    /// ### Step 1: Normalized Coordinate Derivatives
    ///
    /// For x' = x/z and y' = y/z:
    ///
    /// ```text
    /// ∂x'/∂x = 1/z,   ∂x'/∂y = 0,     ∂x'/∂z = -x/z²
    /// ∂y'/∂x = 0,     ∂y'/∂y = 1/z,   ∂y'/∂z = -y/z²
    /// ```
    ///
    /// ### Step 2: Distortion Derivatives (Most Complex)
    ///
    /// The distorted coordinates are: x'' = radial·x' + dx, y'' = radial·y' + dy
    ///
    /// #### Radial Distortion Component
    ///
    /// ```text
    /// radial = 1 + k₁·r² + k₂·r⁴ + k₃·r⁶
    /// ∂radial/∂r² = k₁ + 2k₂·r² + 3k₃·r⁴
    /// ∂r²/∂x' = 2x',  ∂r²/∂y' = 2y'
    /// ```
    ///
    /// #### Tangential Distortion Component
    ///
    /// For dx = 2p₁x'y' + p₂(r² + 2x'²):
    ///
    /// ```text
    /// ∂dx/∂x' = 2p₁y' + p₂(2x' + 4x') = 2p₁y' + 6p₂x'
    /// ∂dx/∂y' = 2p₁x' + 2p₂y'
    /// ```
    ///
    /// For dy = p₁(r² + 2y'²) + 2p₂x'y':
    ///
    /// ```text
    /// ∂dy/∂x' = 2p₁x' + 2p₂y'
    /// ∂dy/∂y' = p₁(2y' + 4y') + 2p₂x' = 6p₁y' + 2p₂x'
    /// ```
    ///
    /// #### Combined Distorted Coordinate Derivatives
    ///
    /// For x'' = radial·x' + dx:
    ///
    /// ```text
    /// ∂x''/∂x' = radial + x'·∂radial/∂r²·∂r²/∂x' + ∂dx/∂x'
    ///          = radial + x'·dradial_dr2·2x' + 2p₁y' + 6p₂x'
    ///          = radial + 2x'²·dradial_dr2 + 2p₁y' + 6p₂x'
    ///
    /// ∂x''/∂y' = x'·∂radial/∂r²·∂r²/∂y' + ∂dx/∂y'
    ///          = x'·dradial_dr2·2y' + 2p₁x' + 2p₂y'
    ///          = 2x'y'·dradial_dr2 + 2p₁x' + 2p₂y'
    /// ```
    ///
    /// For y'' = radial·y' + dy:
    ///
    /// ```text
    /// ∂y''/∂x' = y'·∂radial/∂r²·∂r²/∂x' + ∂dy/∂x'
    ///          = y'·dradial_dr2·2x' + 2p₁x' + 2p₂y'
    ///          = 2x'y'·dradial_dr2 + 2p₁x' + 2p₂y'
    ///
    /// ∂y''/∂y' = radial + y'·∂radial/∂r²·∂r²/∂y' + ∂dy/∂y'
    ///          = radial + y'·dradial_dr2·2y' + 6p₁y' + 2p₂x'
    ///          = radial + 2y'²·dradial_dr2 + 6p₁y' + 2p₂x'
    /// ```
    ///
    /// ### Step 3: Final Projection Derivatives
    ///
    /// For u = fx·x'' + cx and v = fy·y'' + cy:
    ///
    /// ```text
    /// ∂u/∂x'' = fx,  ∂u/∂y'' = 0
    /// ∂v/∂x'' = 0,   ∂v/∂y'' = fy
    /// ```
    ///
    /// ### Full Chain Rule
    ///
    /// ```text
    /// ∂u/∂x = fx·(∂x''/∂x'·∂x'/∂x + ∂x''/∂y'·∂y'/∂x)
    ///       = fx·(∂x''/∂x'·1/z + 0)
    ///       = fx·∂x''/∂x'·1/z
    ///
    /// ∂u/∂y = fx·(∂x''/∂x'·∂x'/∂y + ∂x''/∂y'·∂y'/∂y)
    ///       = fx·(0 + ∂x''/∂y'·1/z)
    ///       = fx·∂x''/∂y'·1/z
    ///
    /// ∂u/∂z = fx·(∂x''/∂x'·∂x'/∂z + ∂x''/∂y'·∂y'/∂z)
    ///       = fx·(∂x''/∂x'·(-x'/z) + ∂x''/∂y'·(-y'/z))
    /// ```
    ///
    /// Similar derivations apply for ∂v/∂x, ∂v/∂y, ∂v/∂z.
    ///
    /// ## Matrix Form
    ///
    /// ```text
    /// J = [ ∂u/∂x  ∂u/∂y  ∂u/∂z ]
    ///     [ ∂v/∂x  ∂v/∂y  ∂v/∂z ]
    /// ```
    ///
    /// ## References
    ///
    /// - Brown, "Decentering Distortion of Lenses", Photogrammetric Engineering 1966
    /// - OpenCV Camera Calibration Documentation
    /// - Hartley & Zisserman, "Multiple View Geometry", Chapter 7
    ///
    /// ## Numerical Verification
    ///
    /// Verified against numerical differentiation in `test_jacobian_point_numerical()`.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_prime = p_cam.x * inv_z;
        let y_prime = p_cam.y * inv_z;

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let (k1, k2, p1, p2, k3) = self
            .distortion_params()
            .expect("RadTanCamera validated at construction");

        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let dradial_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

        // Derivatives of distorted coordinates w.r.t. normalized coordinates
        // x_dist = radial·x' + dx where dx = 2p₁x'y' + p₂(r² + 2x'²)
        // ∂x_dist/∂x' = radial + x'·∂radial/∂r²·∂r²/∂x' + ∂dx/∂x'
        //             = radial + x'·dradial_dr2·2x' + (2p₁y' + p₂·(2x' + 4x'))
        //             = radial + 2x'²·dradial_dr2 + 2p₁y' + 6p₂x'
        let dx_dist_dx_prime = radial
            + 2.0 * x_prime * x_prime * dradial_dr2
            + 2.0 * p1 * y_prime
            + 6.0 * p2 * x_prime;

        // ∂x_dist/∂y' = x'·∂radial/∂r²·∂r²/∂y' + ∂dx/∂y'
        //             = x'·dradial_dr2·2y' + (2p₁x' + 2p₂y')
        let dx_dist_dy_prime =
            2.0 * x_prime * y_prime * dradial_dr2 + 2.0 * p1 * x_prime + 2.0 * p2 * y_prime;

        // y_dist = radial·y' + dy where dy = p₁(r² + 2y'²) + 2p₂x'y'
        // ∂y_dist/∂x' = y'·∂radial/∂r²·∂r²/∂x' + ∂dy/∂x'
        //             = y'·dradial_dr2·2x' + (p₁·2x' + 2p₂y')
        let dy_dist_dx_prime =
            2.0 * y_prime * x_prime * dradial_dr2 + 2.0 * p1 * x_prime + 2.0 * p2 * y_prime;

        // ∂y_dist/∂y' = radial + y'·∂radial/∂r²·∂r²/∂y' + ∂dy/∂y'
        //             = radial + y'·dradial_dr2·2y' + (p₁·(2y' + 4y') + 2p₂x')
        //             = radial + 2y'²·dradial_dr2 + 6p₁y' + 2p₂x'
        let dy_dist_dy_prime = radial
            + 2.0 * y_prime * y_prime * dradial_dr2
            + 6.0 * p1 * y_prime
            + 2.0 * p2 * x_prime;

        // Derivatives of normalized coordinates w.r.t. camera coordinates
        // x' = x/z => ∂x'/∂x = 1/z, ∂x'/∂y = 0, ∂x'/∂z = -x/z²
        // y' = y/z => ∂y'/∂x = 0, ∂y'/∂y = 1/z, ∂y'/∂z = -y/z²

        // Chain rule: ∂(u,v)/∂(x,y,z) = ∂(u,v)/∂(x_dist,y_dist) · ∂(x_dist,y_dist)/∂(x',y') · ∂(x',y')/∂(x,y,z)

        let du_dx = self.pinhole.fx * (dx_dist_dx_prime * inv_z);
        let du_dy = self.pinhole.fx * (dx_dist_dy_prime * inv_z);
        let du_dz = self.pinhole.fx
            * (dx_dist_dx_prime * (-x_prime * inv_z) + dx_dist_dy_prime * (-y_prime * inv_z));

        let dv_dx = self.pinhole.fy * (dy_dist_dx_prime * inv_z);
        let dv_dy = self.pinhole.fy * (dy_dist_dy_prime * inv_z);
        let dv_dz = self.pinhole.fy
            * (dy_dist_dx_prime * (-x_prime * inv_z) + dy_dist_dy_prime * (-y_prime * inv_z));

        SMatrix::<f64, 2, 3>::new(du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz)
    }

    /// Jacobian of projection w.r.t. camera pose (SE3).
    ///
    /// # Mathematical Derivation
    ///
    /// See FovCamera::jacobian_pose() and KannalaBrandtCamera::jacobian_pose() for detailed SE(3) Lie theory derivation.
    ///
    /// ## Summary
    ///
    /// Returns `(J_pixel_point, J_point_pose)` where:
    /// - J_pixel_point: 2×3 Jacobian ∂uv/∂p_cam (from `jacobian_point()`)
    /// - J_point_pose: 3×6 Jacobian ∂p_cam/∂δξ = [ -R^T | [p_cam]× ]
    ///
    /// The full chain is: J_pixel_pose = J_pixel_point · J_point_pose
    ///
    /// ## SE(3) Parameterization
    ///
    /// The SE(3) tangent space has 6 DOF:
    /// - δξ = [δv, δω]^T where δv ∈ ℝ³ (translation), δω ∈ ℝ³ (rotation)
    ///
    /// ## Transformation
    ///
    /// For a world point p_world transformed to camera frame:
    ///
    /// ```text
    /// p_cam = T^{-1} · p_world = R^T · (p_world - t)
    /// ```
    ///
    /// The Jacobian w.r.t. pose perturbation is:
    ///
    /// ```text
    /// J_point_pose = [ -R^T | [p_cam]× ]
    /// ```
    ///
    /// where [p_cam]× is the skew-symmetric matrix of p_cam.
    ///
    /// ## References
    ///
    /// - Barfoot, "State Estimation for Robotics", Chapter 7
    /// - Sola et al., "A micro Lie theory for state estimation in robotics", arXiv:1812.01537
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

    /// Jacobian of projection w.r.t. intrinsic parameters (2×9).
    ///
    /// # Mathematical Derivation
    ///
    /// The Radial-Tangential camera has 9 intrinsic parameters: [fx, fy, cx, cy, k₁, k₂, p₁, p₂, k₃]
    ///
    /// ## Projection Equations
    ///
    /// ```text
    /// x' = x/z,  y' = y/z
    /// r² = x'² + y'²
    /// radial = 1 + k₁·r² + k₂·r⁴ + k₃·r⁶
    /// dx = 2·p₁·x'·y' + p₂·(r² + 2·x'²)
    /// dy = p₁·(r² + 2·y'²) + 2·p₂·x'·y'
    /// x'' = radial·x' + dx
    /// y'' = radial·y' + dy
    /// u = fx·x'' + cx
    /// v = fy·y'' + cy
    /// ```
    ///
    /// ## Jacobian Structure
    ///
    /// ```text
    /// J = [ ∂u/∂fx  ∂u/∂fy  ∂u/∂cx  ∂u/∂cy  ∂u/∂k₁  ∂u/∂k₂  ∂u/∂p₁  ∂u/∂p₂  ∂u/∂k₃ ]
    ///     [ ∂v/∂fx  ∂v/∂fy  ∂v/∂cx  ∂v/∂cy  ∂v/∂k₁  ∂v/∂k₂  ∂v/∂p₁  ∂v/∂p₂  ∂v/∂k₃ ]
    /// ```
    ///
    /// ## Linear Parameters (fx, fy, cx, cy)
    ///
    /// These appear linearly in the projection:
    ///
    /// ```text
    /// ∂u/∂fx = x'',  ∂u/∂fy = 0,   ∂u/∂cx = 1,  ∂u/∂cy = 0
    /// ∂v/∂fx = 0,    ∂v/∂fy = y'', ∂v/∂cx = 0,  ∂v/∂cy = 1
    /// ```
    ///
    /// ## Radial Distortion Coefficients (k₁, k₂, k₃)
    ///
    /// Each k_i affects the radial distortion component:
    ///
    /// ```text
    /// ∂radial/∂k₁ = r²
    /// ∂radial/∂k₂ = r⁴
    /// ∂radial/∂k₃ = r⁶
    /// ```
    ///
    /// By chain rule (x'' = radial·x' + dx, y'' = radial·y' + dy):
    ///
    /// ```text
    /// ∂x''/∂k₁ = x'·r²
    /// ∂x''/∂k₂ = x'·r⁴
    /// ∂x''/∂k₃ = x'·r⁶
    ///
    /// ∂y''/∂k₁ = y'·r²
    /// ∂y''/∂k₂ = y'·r⁴
    /// ∂y''/∂k₃ = y'·r⁶
    /// ```
    ///
    /// Then:
    ///
    /// ```text
    /// ∂u/∂k₁ = fx·x'·r²
    /// ∂u/∂k₂ = fx·x'·r⁴
    /// ∂u/∂k₃ = fx·x'·r⁶
    ///
    /// ∂v/∂k₁ = fy·y'·r²
    /// ∂v/∂k₂ = fy·y'·r⁴
    /// ∂v/∂k₃ = fy·y'·r⁶
    /// ```
    ///
    /// ## Tangential Distortion Coefficients (p₁, p₂)
    ///
    /// For p₁ (affects dx through 2p₁x'y' and dy through p₁(r² + 2y'²)):
    ///
    /// ```text
    /// ∂dx/∂p₁ = 2x'y'
    /// ∂dy/∂p₁ = r² + 2y'²
    /// ```
    ///
    /// Then:
    ///
    /// ```text
    /// ∂u/∂p₁ = fx·∂x''/∂p₁ = fx·2x'y'
    /// ∂v/∂p₁ = fy·∂y''/∂p₁ = fy·(r² + 2y'²)
    /// ```
    ///
    /// For p₂ (affects dx through p₂(r² + 2x'²) and dy through 2p₂x'y'):
    ///
    /// ```text
    /// ∂dx/∂p₂ = r² + 2x'²
    /// ∂dy/∂p₂ = 2x'y'
    /// ```
    ///
    /// Then:
    ///
    /// ```text
    /// ∂u/∂p₂ = fx·∂x''/∂p₂ = fx·(r² + 2x'²)
    /// ∂v/∂p₂ = fy·∂y''/∂p₂ = fy·2x'y'
    /// ```
    ///
    /// ## Matrix Form
    ///
    /// ```text
    /// J = [ x''  0   1  0  fx·x'·r²  fx·x'·r⁴  fx·2x'y'    fx·(r²+2x'²)  fx·x'·r⁶ ]
    ///     [  0  y''  0  1  fy·y'·r²  fy·y'·r⁴  fy·(r²+2y'²) fy·2x'y'      fy·y'·r⁶ ]
    /// ```
    ///
    /// ## References
    ///
    /// - Brown, "Decentering Distortion of Lenses", 1966
    /// - OpenCV Camera Calibration Documentation
    /// - Hartley & Zisserman, "Multiple View Geometry", Chapter 7
    ///
    /// ## Numerical Verification
    ///
    /// Verified in `test_jacobian_intrinsics_numerical()` with tolerance < 1e-5.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let inv_z = 1.0 / p_cam.z;
        let x_prime = p_cam.x * inv_z;
        let y_prime = p_cam.y * inv_z;

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let (k1, k2, p1, p2, k3) = self
            .distortion_params()
            .expect("RadTanCamera validated at construction");

        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        let xy = x_prime * y_prime;
        let dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x_prime * x_prime);
        let dy = p1 * (r2 + 2.0 * y_prime * y_prime) + 2.0 * p2 * xy;

        let x_distorted = radial * x_prime + dx;
        let y_distorted = radial * y_prime + dy;

        // ∂u/∂fx = x_distorted, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = y_distorted, ∂v/∂cx = 0, ∂v/∂cy = 1

        // Distortion parameter derivatives
        let du_dk1 = self.pinhole.fx * x_prime * r2;
        let du_dk2 = self.pinhole.fx * x_prime * r4;
        let du_dp1 = self.pinhole.fx * 2.0 * xy;
        let du_dp2 = self.pinhole.fx * (r2 + 2.0 * x_prime * x_prime);
        let du_dk3 = self.pinhole.fx * x_prime * r6;

        let dv_dk1 = self.pinhole.fy * y_prime * r2;
        let dv_dk2 = self.pinhole.fy * y_prime * r4;
        let dv_dp1 = self.pinhole.fy * (r2 + 2.0 * y_prime * y_prime);
        let dv_dp2 = self.pinhole.fy * 2.0 * xy;
        let dv_dk3 = self.pinhole.fy * y_prime * r6;

        SMatrix::<f64, 2, 9>::from_row_slice(&[
            x_distorted,
            0.0,
            1.0,
            0.0,
            du_dk1,
            du_dk2,
            du_dp1,
            du_dp2,
            du_dk3,
            0.0,
            y_distorted,
            0.0,
            1.0,
            dv_dk1,
            dv_dk2,
            dv_dp1,
            dv_dp2,
            dv_dk3,
        ])
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - cx, cy must be finite
    /// - k₁, k₂, p₁, p₂, k₃ must be finite
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.pinhole.fx <= 0.0 || self.pinhole.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }

        if !self.pinhole.cx.is_finite() || !self.pinhole.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }

        let (k1, k2, p1, p2, k3) = self.distortion_params()?;
        if !k1.is_finite()
            || !k2.is_finite()
            || !p1.is_finite()
            || !p2.is_finite()
            || !k3.is_finite()
        {
            return Err(CameraModelError::InvalidParams(
                "Distortion coefficients must be finite".to_string(),
            ));
        }

        Ok(())
    }

    fn get_pinhole_params(&self) -> PinholeParams {
        self.pinhole
    }

    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    fn get_model_name(&self) -> &'static str {
        "rad_tan"
    }
}

// ============================================================================
// From/Into Trait Implementations for RadTanCamera
// ============================================================================

/// Convert camera to dynamic vector of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
impl From<&RadTanCamera> for DVector<f64> {
    fn from(camera: &RadTanCamera) -> Self {
        let (k1, k2, p1, p2, k3) = camera
            .distortion_params()
            .expect("RadTanCamera validated at construction");
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            k1,
            k2,
            p1,
            p2,
            k3,
        ])
    }
}

/// Convert camera to fixed-size array of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
impl From<&RadTanCamera> for [f64; 9] {
    fn from(camera: &RadTanCamera) -> Self {
        let (k1, k2, p1, p2, k3) = camera
            .distortion_params()
            .expect("RadTanCamera validated at construction");
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            k1,
            k2,
            p1,
            p2,
            k3,
        ]
    }
}

/// Create camera from slice of intrinsic parameters.
///
/// # Layout
///
/// Expected parameter order: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
///
/// # Panics
///
/// Panics if the slice has fewer than 9 elements.
impl From<&[f64]> for RadTanCamera {
    fn from(params: &[f64]) -> Self {
        assert!(
            params.len() >= 9,
            "RadTanCamera requires at least 9 parameters, got {}",
            params.len()
        );
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::BrownConrady {
                k1: params[4],
                k2: params[5],
                p1: params[6],
                p2: params[7],
                k3: params[8],
            },
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
/// Expected parameter order: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
impl From<[f64; 9]> for RadTanCamera {
    fn from(params: [f64; 9]) -> Self {
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::BrownConrady {
                k1: params[4],
                k2: params[5],
                p1: params[6],
                p2: params[7],
                k3: params[8],
            },
            resolution: Resolution {
                width: 0,
                height: 0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_radtan_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = RadTanCamera::new(pinhole, distortion, resolution)?;
        assert_eq!(camera.pinhole.fx, 300.0);
        let (k1, _, p1, _, _) = camera.distortion_params()?;
        assert_eq!(k1, 0.1);
        assert_eq!(p1, 0.001);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = RadTanCamera::new(pinhole, distortion, resolution)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;

        assert!((uv.x - 320.0).abs() < crate::PROJECTION_TEST_TOLERANCE);
        assert!((uv.y - 240.0).abs() < crate::PROJECTION_TEST_TOLERANCE);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = RadTanCamera::new(pinhole, distortion, resolution)?;
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
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = RadTanCamera::new(pinhole, distortion, resolution)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params: DVector<f64> = (&camera).into();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..9 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = RadTanCamera::from(params_plus.as_slice());
            let cam_minus = RadTanCamera::from(params_minus.as_slice());

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
    fn test_rad_tan_from_into_traits() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::BrownConrady {
            k1: 0.1,
            k2: 0.01,
            p1: 0.001,
            p2: 0.002,
            k3: 0.001,
        };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = RadTanCamera::new(pinhole, distortion, resolution)?;

        // Test conversion to DVector
        let params: DVector<f64> = (&camera).into();
        assert_eq!(params.len(), 9);
        assert_eq!(params[0], 300.0);
        assert_eq!(params[1], 300.0);
        assert_eq!(params[2], 320.0);
        assert_eq!(params[3], 240.0);
        assert_eq!(params[4], 0.1);
        assert_eq!(params[5], 0.01);
        assert_eq!(params[6], 0.001);
        assert_eq!(params[7], 0.002);
        assert_eq!(params[8], 0.001);

        // Test conversion to array
        let arr: [f64; 9] = (&camera).into();
        assert_eq!(
            arr,
            [300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.002, 0.001]
        );

        // Test conversion from slice
        let params_slice = [350.0, 350.0, 330.0, 250.0, 0.2, 0.02, 0.002, 0.003, 0.002];
        let camera2 = RadTanCamera::from(&params_slice[..]);
        assert_eq!(camera2.pinhole.fx, 350.0);
        assert_eq!(camera2.pinhole.fy, 350.0);
        assert_eq!(camera2.pinhole.cx, 330.0);
        assert_eq!(camera2.pinhole.cy, 250.0);
        let (k1, k2, p1, p2, k3) = camera2.distortion_params()?;
        assert_eq!(k1, 0.2);
        assert_eq!(k2, 0.02);
        assert_eq!(p1, 0.002);
        assert_eq!(p2, 0.003);
        assert_eq!(k3, 0.002);

        // Test conversion from array
        let camera3 =
            RadTanCamera::from([400.0, 400.0, 340.0, 260.0, 0.3, 0.03, 0.003, 0.004, 0.003]);
        assert_eq!(camera3.pinhole.fx, 400.0);
        assert_eq!(camera3.pinhole.fy, 400.0);
        let (k1, k2, p1, p2, k3) = camera3.distortion_params()?;
        assert_eq!(k1, 0.3);
        assert_eq!(k2, 0.03);
        assert_eq!(p1, 0.003);
        assert_eq!(p2, 0.004);
        assert_eq!(k3, 0.003);

        Ok(())
    }
}
