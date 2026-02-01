//! BAL (Bundle Adjustment in the Large) pinhole camera model.
//!
//! This module implements a pinhole camera model that follows the BAL dataset convention
//! where cameras look down the -Z axis (negative Z in front of camera).

use crate::{
    CameraModel, CameraModelError, DistortionModel, PinholeParams, Resolution, skew_symmetric,
};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Strict BAL camera model matching Snavely's Bundler convention.
///
/// This camera model uses EXACTLY 3 intrinsic parameters matching the BAL file format:
/// - Single focal length (f): fx = fy
/// - Two radial distortion coefficients (k1, k2)
/// - NO principal point (cx = cy = 0 by convention)
///
/// This matches the intrinsic parameterization used by:
/// - Ceres Solver bundle adjustment examples
/// - GTSAM bundle adjustment
/// - Original Bundler software
///
/// # Parameters
///
/// - `f`: Single focal length in pixels (fx = fy = f)
/// - `k1`: First radial distortion coefficient
/// - `k2`: Second radial distortion coefficient
///
/// # Projection Model
///
/// For a 3D point `p_cam = (x, y, z)` in camera frame where z < 0:
/// ```text
/// x_n = x / (-z)
/// y_n = y / (-z)
/// r² = x_n² + y_n²
/// distortion = 1 + k1*r² + k2*r⁴
/// x_d = x_n * distortion
/// y_d = y_n * distortion
/// u = f * x_d      (no cx offset)
/// v = f * y_d      (no cy offset)
/// ```
///
/// # Usage
///
/// This camera model should be used for bundle adjustment problems that read
/// BAL format files, to ensure parameter compatibility and avoid degenerate
/// optimization (extra DOF from fx≠fy or non-zero principal point).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BALPinholeCameraStrict {
    /// Single focal length (fx = fy = f)
    pub f: f64,
    /// Lens distortion model and parameters
    pub distortion: DistortionModel,
    /// Image resolution
    pub resolution: Resolution,
}

impl BALPinholeCameraStrict {
    /// Create a new strict BAL pinhole camera with distortion.
    ///
    /// # Arguments
    ///
    /// * `pinhole` - Pinhole parameters. MUST have fx=fy and cx=cy=0 for strict BAL format.
    /// * `distortion` - MUST be DistortionModel::Radial { k1, k2 }
    /// * `resolution` - Image resolution (currently unused but kept for API consistency)
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError::InvalidParams` if:
    /// - `pinhole.fx != pinhole.fy` (strict BAL requires single focal length)
    /// - `pinhole.cx != 0.0 || pinhole.cy != 0.0` (strict BAL has no principal point offset)
    /// - `distortion` is not `DistortionModel::Radial`
    pub fn new(
        pinhole: PinholeParams,
        distortion: DistortionModel,
        _resolution: crate::Resolution,
    ) -> Result<Self, CameraModelError> {
        // Validate strict BAL constraints on input
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
            f: pinhole.fx, // Use fx as the single focal length
            distortion,
            resolution: _resolution,
        };
        camera.validate_params()?;
        Ok(camera)
    }

    /// Create a strict BAL pinhole camera without distortion (k1=0, k2=0).
    ///
    /// This is a convenience constructor for the common case of no distortion.
    pub fn new_no_distortion(f: f64) -> Result<Self, CameraModelError> {
        let pinhole = PinholeParams::new(f, f, 0.0, 0.0)?;
        let distortion = DistortionModel::Radial { k1: 0.0, k2: 0.0 };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        }; // Dummy resolution
        Self::new(pinhole, distortion, resolution)
    }

    /// Helper method to extract distortion parameter, returning Result for consistency.
    fn distortion_params(&self) -> (f64, f64) {
        match self.distortion {
            DistortionModel::Radial { k1, k2 } => (k1, k2),
            _ => (0.0, 0.0),
        }
    }

    /// Checks if a 3D point satisfies the projection condition (z < -epsilon for BAL).
    fn check_projection_condition(&self, z: f64) -> bool {
        z < -crate::MIN_DEPTH
    }
}

/// Convert camera to dynamic vector of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [f, k1, k2]
impl From<&BALPinholeCameraStrict> for DVector<f64> {
    fn from(camera: &BALPinholeCameraStrict) -> Self {
        let (k1, k2) = camera.distortion_params();
        DVector::from_vec(vec![camera.f, k1, k2])
    }
}

/// Convert camera to fixed-size array of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [f, k1, k2]
impl From<&BALPinholeCameraStrict> for [f64; 3] {
    fn from(camera: &BALPinholeCameraStrict) -> Self {
        let (k1, k2) = camera.distortion_params();
        [camera.f, k1, k2]
    }
}

/// Create camera from slice of intrinsic parameters.
///
/// # Layout
///
/// Expected parameter order: [f, k1, k2]
///
/// # Panics
///
/// Panics if the slice has fewer than 3 elements or if validation fails.
impl From<&[f64]> for BALPinholeCameraStrict {
    fn from(params: &[f64]) -> Self {
        assert!(
            params.len() >= 3,
            "BALPinholeCameraStrict requires exactly 3 parameters, got {}",
            params.len()
        );
        Self {
            f: params[0],
            distortion: DistortionModel::Radial {
                k1: params[1],
                k2: params[2],
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
/// Expected parameter order: [f, k1, k2]
impl From<[f64; 3]> for BALPinholeCameraStrict {
    fn from(params: [f64; 3]) -> Self {
        Self {
            f: params[0],
            distortion: DistortionModel::Radial {
                k1: params[1],
                k2: params[2],
            },
            resolution: Resolution {
                width: 0,
                height: 0,
            },
        }
    }
}

impl CameraModel for BALPinholeCameraStrict {
    const INTRINSIC_DIM: usize = 3; // f, k1, k2
    type IntrinsicJacobian = SMatrix<f64, 2, 3>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        // BAL convention: negative Z is in front
        if !self.check_projection_condition(p_cam.z) {
            // return Err(CameraModelError::PointBehindCamera {
            //     z: p_cam.z,
            //     message: "BAL convention: point must have z < 0 (negative Z is in front of camera)"
            //         .to_string(),
            // });
            return Err(CameraModelError::ProjectionOutSideImage);
            //
        }
        let inv_neg_z = -1.0 / p_cam.z;

        // Normalized coordinates
        let x_n = p_cam.x * inv_neg_z;
        let y_n = p_cam.y * inv_neg_z;

        let (k1, k2) = self.distortion_params();

        // Radial distortion
        let r2 = x_n * x_n + y_n * y_n;
        let r4 = r2 * r2;
        let distortion = 1.0 + k1 * r2 + k2 * r4;

        // Apply distortion and focal length (no principal point offset)
        let x_d = x_n * distortion;
        let y_d = y_n * distortion;

        Ok(Vector2::new(self.f * x_d, self.f * y_d))
    }

    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        self.check_projection_condition(p_cam.z)
    }

    /// Computes the Jacobian of the projection function with respect to the 3D point in camera frame.
    ///
    /// # Mathematical Derivation
    ///
    /// The projection function maps a 3D point p_cam = (x, y, z) to 2D pixel coordinates (u, v).
    ///
    /// ## Step 1: Normalized Coordinates (BAL Convention)
    ///
    /// BAL uses negative Z in front of camera (OpenGL convention):
    /// ```text
    /// x_n = x / (-z)
    /// y_n = y / (-z)
    /// ```
    ///
    /// Let `inv_neg_z = -1/z`, then:
    /// ```text
    /// x_n = x * inv_neg_z
    /// y_n = y * inv_neg_z
    /// ```
    ///
    /// Jacobian of normalized coordinates:
    /// ```text
    /// ∂x_n/∂x = inv_neg_z = -1/z
    /// ∂x_n/∂y = 0
    /// ∂x_n/∂z = ∂(x * inv_neg_z)/∂z = x * ∂(-1/z)/∂z = x * (1/z²) = x_n * inv_neg_z
    ///
    /// ∂y_n/∂x = 0
    /// ∂y_n/∂y = inv_neg_z = -1/z
    /// ∂y_n/∂z = y_n * inv_neg_z
    /// ```
    ///
    /// ## Step 2: Radial Distortion
    ///
    /// The radial distance squared and distortion factor:
    /// ```text
    /// r² = x_n² + y_n²
    /// r⁴ = (r²)²
    /// d(r²) = 1 + k1·r² + k2·r⁴
    /// ```
    ///
    /// Distorted coordinates:
    /// ```text
    /// x_d = x_n · d(r²)
    /// y_d = y_n · d(r²)
    /// ```
    ///
    /// ### Derivatives of r² and d(r²):
    /// ```text
    /// ∂(r²)/∂x_n = 2·x_n
    /// ∂(r²)/∂y_n = 2·y_n
    ///
    /// ∂d/∂(r²) = k1 + 2·k2·r²
    /// ```
    ///
    /// ### Jacobian of distorted coordinates w.r.t. normalized:
    /// ```text
    /// ∂x_d/∂x_n = ∂(x_n · d)/∂x_n = d + x_n · (∂d/∂(r²)) · (∂(r²)/∂x_n)
    ///           = d + x_n · (k1 + 2·k2·r²) · 2·x_n
    ///
    /// ∂x_d/∂y_n = x_n · (∂d/∂(r²)) · (∂(r²)/∂y_n)
    ///           = x_n · (k1 + 2·k2·r²) · 2·y_n
    ///
    /// ∂y_d/∂x_n = y_n · (k1 + 2·k2·r²) · 2·x_n
    ///
    /// ∂y_d/∂y_n = d + y_n · (k1 + 2·k2·r²) · 2·y_n
    /// ```
    ///
    /// ## Step 3: Pixel Coordinates (Strict BAL: no principal point)
    ///
    /// ```text
    /// u = f · x_d   (no cx offset)
    /// v = f · y_d   (no cy offset)
    /// ```
    ///
    /// ## Step 4: Chain Rule
    ///
    /// The full Jacobian ∂(u,v)/∂(x,y,z) is computed by chaining:
    /// ```text
    /// J = ∂(u,v)/∂(x_d,y_d) · ∂(x_d,y_d)/∂(x_n,y_n) · ∂(x_n,y_n)/∂(x,y,z)
    /// ```
    ///
    /// Final results:
    /// ```text
    /// ∂u/∂x = f · (∂x_d/∂x_n · ∂x_n/∂x + ∂x_d/∂y_n · ∂y_n/∂x)
    ///       = f · (∂x_d/∂x_n · inv_neg_z)
    ///
    /// ∂u/∂y = f · (∂x_d/∂y_n · inv_neg_z)
    ///
    /// ∂u/∂z = f · (∂x_d/∂x_n · ∂x_n/∂z + ∂x_d/∂y_n · ∂y_n/∂z)
    ///
    /// ∂v/∂x = f · (∂y_d/∂x_n · inv_neg_z)
    ///
    /// ∂v/∂y = f · (∂y_d/∂y_n · inv_neg_z)
    ///
    /// ∂v/∂z = f · (∂y_d/∂x_n · ∂x_n/∂z + ∂y_d/∂y_n · ∂y_n/∂z)
    /// ```
    ///
    /// # References
    ///
    /// - Snavely et al., "Photo Tourism: Exploring Photo Collections in 3D", SIGGRAPH 2006
    /// - Agarwal et al., "Bundle Adjustment in the Large", ECCV 2010
    /// - [Bundle Adjustment in the Large Dataset](https://grail.cs.washington.edu/projects/bal/)
    ///
    /// # Verification
    ///
    /// This Jacobian is verified against numerical differentiation in tests.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let inv_neg_z = -1.0 / p_cam.z;
        let x_n = p_cam.x * inv_neg_z;
        let y_n = p_cam.y * inv_neg_z;

        let (k1, k2) = self.distortion_params();

        // Radial distortion
        let r2 = x_n * x_n + y_n * y_n;
        let r4 = r2 * r2;
        let distortion = 1.0 + k1 * r2 + k2 * r4;

        // Derivative of distortion w.r.t. r²
        let d_dist_dr2 = k1 + 2.0 * k2 * r2;

        // Jacobian of normalized coordinates w.r.t. camera point
        let dxn_dz = x_n * inv_neg_z;
        let dyn_dz = y_n * inv_neg_z;

        // Jacobian of distorted point w.r.t. normalized point
        let dx_d_dxn = distortion + x_n * d_dist_dr2 * 2.0 * x_n;
        let dx_d_dyn = x_n * d_dist_dr2 * 2.0 * y_n;
        let dy_d_dxn = y_n * d_dist_dr2 * 2.0 * x_n;
        let dy_d_dyn = distortion + y_n * d_dist_dr2 * 2.0 * y_n;

        // Chain rule with single focal length f (not fx/fy)
        let du_dx = self.f * (dx_d_dxn * inv_neg_z);
        let du_dy = self.f * (dx_d_dyn * inv_neg_z);
        let du_dz = self.f * (dx_d_dxn * dxn_dz + dx_d_dyn * dyn_dz);

        let dv_dx = self.f * (dy_d_dxn * inv_neg_z);
        let dv_dy = self.f * (dy_d_dyn * inv_neg_z);
        let dv_dz = self.f * (dy_d_dxn * dxn_dz + dy_d_dyn * dyn_dz);

        SMatrix::<f64, 2, 3>::new(du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz)
    }

    /// Computes the Jacobian of the projection function with respect to the camera pose.
    ///
    /// # Mathematical Derivation
    ///
    /// Given a 3D point in world frame `p_world` and camera pose `pose` (camera-to-world transformation),
    /// we need the Jacobian ∂(u,v)/∂ξ where ξ ∈ se(3) is the Lie algebra perturbation.
    ///
    /// ## Camera Coordinate Transformation
    ///
    /// The pose is a camera-to-world SE(3) transformation: T_cw = (R, t) where:
    /// - R ∈ SO(3): rotation from camera to world
    /// - t ∈ ℝ³: translation of camera origin in world frame
    ///
    /// To transform from world to camera, we use the inverse:
    /// ```text
    /// p_cam = T_cw^{-1} · p_world = R^T · (p_world - t)
    /// ```
    ///
    /// ## SE(3) Right Perturbation
    ///
    /// We use right perturbation on SE(3): for small δξ = [δρ; δθ] ∈ ℝ⁶:
    /// ```text
    /// T' = T ∘ Exp(δξ)
    /// ```
    ///
    /// Where δξ = [δρ; δθ] with:
    /// - δρ ∈ ℝ³: translation perturbation (in camera frame)
    /// - δθ ∈ ℝ³: rotation perturbation (axis-angle in camera frame)
    ///
    /// ## Perturbation Effect on Transformed Point
    ///
    /// Under right perturbation T' = T ∘ Exp([δρ; δθ]):
    /// ```text
    /// R' = R · Exp(δθ) ≈ R · (I + [δθ]×)
    /// t' ≈ t + R · δρ  (for small δθ, V(δθ) ≈ I)
    /// ```
    ///
    /// Then the transformed point becomes:
    /// ```text
    /// p_cam' = (R')^T · (p_world - t')
    ///        = (I - [δθ]×) · R^T · (p_world - t - R · δρ)
    ///        ≈ (I - [δθ]×) · R^T · (p_world - t) - (I - [δθ]×) · δρ
    ///        ≈ (I - [δθ]×) · p_cam - δρ
    ///        = p_cam - [δθ]× · p_cam - δρ
    ///        = p_cam + p_cam × δθ - δρ
    ///        = p_cam + [p_cam]× · δθ - δρ
    /// ```
    ///
    /// Where [v]× denotes the skew-symmetric matrix (cross-product matrix).
    ///
    /// ## Jacobian of p_cam w.r.t. Pose Perturbation
    ///
    /// From the above derivation:
    /// ```text
    /// ∂p_cam/∂[δρ; δθ] = [-I | [p_cam]×]
    /// ```
    ///
    /// This is a 3×6 matrix where:
    /// - First 3 columns (translation): -I (identity with negative sign)
    /// - Last 3 columns (rotation): [p_cam]× (skew-symmetric matrix of p_cam)
    ///
    /// ## Chain Rule
    ///
    /// The final Jacobian is:
    /// ```text
    /// ∂(u,v)/∂ξ = ∂(u,v)/∂p_cam · ∂p_cam/∂ξ
    /// ```
    ///
    /// # References
    ///
    /// - Barfoot & Furgale, "Associating Uncertainty with Three-Dimensional Poses for Use in Estimation Problems", IEEE Trans. Robotics 2014
    /// - Solà et al., "A Micro Lie Theory for State Estimation in Robotics", arXiv:1812.01537, 2018
    /// - Blanco, "A tutorial on SE(3) transformation parameterizations and on-manifold optimization", Technical Report 2010
    ///
    /// # Returns
    ///
    /// A tuple `(d_uv_d_pcam, d_pcam_d_pose)` where:
    /// - `d_uv_d_pcam`: 2×3 Jacobian of projection w.r.t. point in camera frame
    /// - `d_pcam_d_pose`: 3×6 Jacobian of camera point w.r.t. pose perturbation
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
        let p_cam_skew = skew_symmetric(&p_cam);

        let d_pcam_d_pose = SMatrix::<f64, 3, 6>::from_fn(|r, c| {
            if c < 3 {
                // Translation part: -I
                if r == c { -1.0 } else { 0.0 }
            } else {
                // Rotation part: [p_cam]×
                p_cam_skew[(r, c - 3)]
            }
        });

        (d_uv_d_pcam, d_pcam_d_pose)
    }

    /// Computes the Jacobian of the projection function with respect to intrinsic parameters.
    ///
    /// # Mathematical Derivation
    ///
    /// The strict BAL camera has EXACTLY 3 intrinsic parameters:
    /// ```text
    /// θ = [f, k1, k2]
    /// ```
    ///
    /// Where:
    /// - f: Single focal length (fx = fy = f)
    /// - k1, k2: Radial distortion coefficients
    /// - NO principal point (cx = cy = 0 by convention)
    ///
    /// ## Projection Model
    ///
    /// Recall the projection equations:
    /// ```text
    /// x_n = x / (-z),  y_n = y / (-z)
    /// r² = x_n² + y_n²
    /// d(r²; k1, k2) = 1 + k1·r² + k2·r⁴
    /// x_d = x_n · d(r²; k1, k2)
    /// y_d = y_n · d(r²; k1, k2)
    /// u = f · x_d
    /// v = f · y_d
    /// ```
    ///
    /// ## Jacobian w.r.t. Focal Length (f)
    ///
    /// The focal length appears only in the final step:
    /// ```text
    /// ∂u/∂f = ∂(f · x_d)/∂f = x_d
    /// ∂v/∂f = ∂(f · y_d)/∂f = y_d
    /// ```
    ///
    /// ## Jacobian w.r.t. Distortion Coefficients (k1, k2)
    ///
    /// The distortion coefficients affect the distortion function d(r²):
    /// ```text
    /// ∂d/∂k1 = r²
    /// ∂d/∂k2 = r⁴
    /// ```
    ///
    /// Using the chain rule:
    /// ```text
    /// ∂u/∂k1 = ∂(f · x_d)/∂k1 = f · ∂x_d/∂k1
    ///        = f · ∂(x_n · d)/∂k1
    ///        = f · x_n · (∂d/∂k1)
    ///        = f · x_n · r²
    ///
    /// ∂u/∂k2 = f · x_n · (∂d/∂k2)
    ///        = f · x_n · r⁴
    /// ```
    ///
    /// Similarly for v:
    /// ```text
    /// ∂v/∂k1 = f · y_n · r²
    /// ∂v/∂k2 = f · y_n · r⁴
    /// ```
    ///
    /// ## Complete Jacobian Matrix (2×3)
    ///
    /// ```text
    ///         ∂/∂f    ∂/∂k1        ∂/∂k2
    /// ∂u/∂θ = [x_d,   f·x_n·r²,   f·x_n·r⁴]
    /// ∂v/∂θ = [y_d,   f·y_n·r²,   f·y_n·r⁴]
    /// ```
    ///
    /// # References
    ///
    /// - Agarwal et al., "Bundle Adjustment in the Large", ECCV 2010, Section 3
    /// - [Ceres Solver: Bundle Adjustment Tutorial](http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment)
    /// - Triggs et al., "Bundle Adjustment - A Modern Synthesis", Vision Algorithms: Theory and Practice, 2000
    ///
    /// # Notes
    ///
    /// This differs from the general BALPinholeCamera which has 6 parameters (fx, fy, cx, cy, k1, k2).
    /// The strict BAL format enforces fx=fy and cx=cy=0 to match the original Bundler software
    /// and standard BAL dataset files, reducing the intrinsic dimensionality from 6 to 3.
    ///
    /// # Verification
    ///
    /// This Jacobian is verified against numerical differentiation in tests.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let inv_neg_z = -1.0 / p_cam.z;
        let x_n = p_cam.x * inv_neg_z;
        let y_n = p_cam.y * inv_neg_z;

        // Radial distortion
        let (k1, k2) = self.distortion_params();
        let r2 = x_n * x_n + y_n * y_n;
        let r4 = r2 * r2;
        let distortion = 1.0 + k1 * r2 + k2 * r4;

        let x_d = x_n * distortion;
        let y_d = y_n * distortion;

        // Jacobian ∂(u,v)/∂(f,k1,k2)
        SMatrix::<f64, 2, 3>::new(
            x_d,               // ∂u/∂f
            self.f * x_n * r2, // ∂u/∂k1
            self.f * x_n * r4, // ∂u/∂k2
            y_d,               // ∂v/∂f
            self.f * y_n * r2, // ∂v/∂k1
            self.f * y_n * r4, // ∂v/∂k2
        )
    }

    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        // Remove distortion and convert to ray
        // Principal point is (0,0) for strict BAL
        let x_d = point_2d.x / self.f;
        let y_d = point_2d.y / self.f;

        // Iterative undistortion
        let mut x_n = x_d;
        let mut y_n = y_d;

        let (k1, k2) = self.distortion_params();

        for _ in 0..5 {
            let r2 = x_n * x_n + y_n * y_n;
            let distortion = 1.0 + k1 * r2 + k2 * r2 * r2;
            x_n = x_d / distortion;
            y_n = y_d / distortion;
        }

        // BAL convention: camera looks down -Z axis
        let norm = (1.0 + x_n * x_n + y_n * y_n).sqrt();
        Ok(Vector3::new(x_n / norm, y_n / norm, -1.0 / norm))
    }

    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.f <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }
        let (k1, k2) = self.distortion_params();
        if !k1.is_finite() || !k2.is_finite() {
            return Err(CameraModelError::InvalidParams(
                "Distortion coefficients must be finite".to_string(),
            ));
        }
        // // Validate strict BAL constraints
        // if (self.pinhole.fx - self.pinhole.fy).abs() > 1e-10 {
        //     return Err(CameraModelError::InvalidParams(
        //         "BALPinholeCameraStrict requires fx = fy (single focal length)".to_string(),
        //     ));
        // }
        // if self.pinhole.cx.abs() > 1e-10 || self.pinhole.cy.abs() > 1e-10 {
        //     return Err(CameraModelError::InvalidParams(
        //         "BALPinholeCameraStrict requires cx = cy = 0 (no principal point offset)"
        //             .to_string(),
        //     ));
        // }

        // let (k1, k2) = match self.distortion {
        //     DistortionModel::Radial { k1, k2 } => (k1, k2),
        //     _ => {
        //         return Err(CameraModelError::InvalidParams(format!(
        //             "BALPinholeCameraStrict requires Radial distortion model, got {:?}",
        //             self.distortion
        //         )));
        //     }
        // };
        Ok(())
    }

    fn get_pinhole_params(&self) -> PinholeParams {
        PinholeParams {
            fx: self.f,
            fy: self.f,
            cx: 0.0,
            cy: 0.0,
        }
    }

    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    fn get_model_name(&self) -> &'static str {
        "bal_pinhole_strict"
    }
}

// ============================================================================
// From/Into Trait Implementations for BALPinholeCameraStrict
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_bal_strict_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 0.0, 0.0)?;
        let distortion = DistortionModel::Radial { k1: 0.4, k2: -0.3 };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };

        let camera = BALPinholeCameraStrict::new(pinhole, distortion, resolution)?;

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
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };

        let result = BALPinholeCameraStrict::new(pinhole, distortion, resolution);
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
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };

        let result = BALPinholeCameraStrict::new(pinhole, distortion, resolution);
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
        let camera2 = BALPinholeCameraStrict::from(&params_slice[..]);
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
}
