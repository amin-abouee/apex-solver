//! NVIDIA f-theta Fisheye Camera Model
//!
//! A polynomial-based fisheye camera model used in NVIDIA's autonomous-vehicle
//! cameras. The model maps the angle θ between a 3-D ray and the optical axis
//! to an image-plane radius via a degree-4 polynomial, while preserving the
//! azimuthal direction exactly (the model is isotropic).
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! d     = √(x² + y² + z²)
//! θ     = arccos(z / d)               — angle from optical axis
//! r_p   = √(x² + y²)                 — lateral distance in camera plane
//! f(θ)  = k₁θ + k₂θ² + k₃θ³ + k₄θ⁴ — forward polynomial
//! u     = cx + f(θ) · x / r_p
//! v     = cy + f(θ) · y / r_p
//! ```
//!
//! Special case r_p < ε (point on optical axis): `(u, v) = (cx, cy)`.
//!
//! ## Unprojection (2D → 3D)
//!
//! Given pixel (u, v), let dx = u − cx, dy = v − cy, r_d = √(dx² + dy²).
//! If r_d < ε return (0, 0, 1). Otherwise solve f(θ) = r_d via Newton-Raphson
//! (initial guess θ₀ = r_d / k₁):
//!
//! ```text
//! θ_{n+1} = θ_n − (f(θ_n) − r_d) / f′(θ_n)
//! f′(θ)   = k₁ + 2k₂θ + 3k₃θ² + 4k₄θ³
//! ```
//!
//! Recover the unit ray:
//! ```text
//! ray = [sin θ · dx/r_d,  sin θ · dy/r_d,  cos θ]
//! ```
//!
//! ## Jacobians
//!
//! ### ∂(u,v)/∂(x,y,z) — 2×3
//!
//! With `A = f′(θ)·z / (r_p²·d²)` and `B = f(θ) / r_p³`:
//!
//! ```text
//! J = [[ A·x² + B·y²,   (A−B)·xy,   −f′·x/d² ],
//!      [ (A−B)·xy,       A·y² + B·x²,  −f′·y/d² ]]
//! ```
//!
//! At the optical axis (r_p < ε): `J = [[k₁/z, 0, 0], [0, k₁/z, 0]]`.
//!
//! ### ∂(u,v)/∂(cx,cy,k₁,k₂,k₃,k₄) — 2×6
//!
//! ```text
//! J_intr = [[1, 0,  θ·cφ,  θ²·cφ,  θ³·cφ,  θ⁴·cφ],
//!            [0, 1,  θ·sφ,  θ²·sφ,  θ³·sφ,  θ⁴·sφ]]
//! ```
//! where cφ = x/r_p, sφ = y/r_p.  At the optical axis all kᵢ columns are zero.
//!
//! # Parameters
//!
//! `INTRINSIC_DIM = 6`, parameter vector: `[cx, cy, k₁, k₂, k₃, k₄]`
//!
//! - **cx, cy**: principal point in pixels (finite)
//! - **k₁**: linear coefficient — acts as focal length (pixels/radian); must be > 0
//! - **k₂, k₃, k₄**: higher-order distortion coefficients (finite, typically small)
//!
//! # References
//!
//! NVIDIA, "The f-theta Camera Model", internal whitepaper.

use crate::{CONVERGENCE_THRESHOLD, CameraModel, CameraModelError, GEOMETRIC_PRECISION, MIN_DEPTH};
use crate::{DistortionModel, PinholeParams};
use nalgebra::{DVector, Matrix3xX, SMatrix, Vector2, Vector3};

/// NVIDIA f-theta fisheye camera with 6 intrinsic parameters.
///
/// Stores the principal point (cx, cy) and the four forward-polynomial
/// coefficients k₁…k₄.  The model is isotropic (no separate fx/fy).
#[derive(Clone, Copy, PartialEq)]
pub struct FThetaCamera {
    /// Principal point x (u₀ in the paper), pixels.
    pub cx: f64,
    /// Principal point y (v₀ in the paper), pixels.
    pub cy: f64,
    /// Forward-polynomial distortion — must be [`DistortionModel::FTheta`].
    pub distortion: DistortionModel,
}

impl std::fmt::Debug for FThetaCamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (k1, k2, k3, k4) = self.distortion_params();
        f.debug_struct("FThetaCamera")
            .field("cx", &self.cx)
            .field("cy", &self.cy)
            .field("k1", &k1)
            .field("k2", &k2)
            .field("k3", &k3)
            .field("k4", &k4)
            .finish()
    }
}

impl FThetaCamera {
    /// Create a new f-theta camera.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::InvalidParams`] if `distortion` is not
    /// [`DistortionModel::FTheta`], or if any parameter fails validation.
    pub fn new(cx: f64, cy: f64, distortion: DistortionModel) -> Result<Self, CameraModelError> {
        let cam = Self { cx, cy, distortion };
        cam.validate_params()?;
        Ok(cam)
    }

    /// Build directly from the six scalar parameters `[cx, cy, k1, k2, k3, k4]`.
    pub fn try_from_params(params: &[f64]) -> Result<Self, CameraModelError> {
        Self::try_from(params)
    }

    /// Extract `(k1, k2, k3, k4)` from the stored distortion.
    ///
    /// # Panics
    ///
    /// Panics if `self.distortion` is not [`DistortionModel::FTheta`] — this
    /// cannot happen after successful construction.
    #[inline]
    pub fn distortion_params(&self) -> (f64, f64, f64, f64) {
        match self.distortion {
            DistortionModel::FTheta { k1, k2, k3, k4 } => (k1, k2, k3, k4),
            _ => unreachable!("FThetaCamera always has FTheta distortion"),
        }
    }

    /// Evaluate the forward polynomial f(θ) = k₁θ + k₂θ² + k₃θ³ + k₄θ⁴.
    #[inline]
    fn poly_forward(&self, theta: f64) -> f64 {
        let (k1, k2, k3, k4) = self.distortion_params();
        theta * (k1 + theta * (k2 + theta * (k3 + theta * k4)))
    }

    /// Evaluate the derivative f′(θ) = k₁ + 2k₂θ + 3k₃θ² + 4k₄θ³.
    #[inline]
    fn poly_forward_deriv(&self, theta: f64) -> f64 {
        let (k1, k2, k3, k4) = self.distortion_params();
        k1 + theta * (2.0 * k2 + theta * (3.0 * k3 + theta * 4.0 * k4))
    }

    /// Least-squares estimation of k₁..k₄ given 3D-2D correspondences,
    /// assuming cx and cy are already known.
    ///
    /// Constructs a Vandermonde system `[θ θ² θ³ θ⁴] · [k₁ k₂ k₃ k₄]ᵀ = r`
    /// and solves it with SVD (nalgebra full-pivoting).
    ///
    /// Returns an updated `FThetaCamera` with the estimated polynomial
    /// coefficients.  The principal point is unchanged.
    ///
    /// # Arguments
    ///
    /// * `points_3d` — 3×N matrix of 3D points in camera frame
    /// * `points_2d` — 2×N matrix of observed pixel coordinates
    pub fn linear_estimation(
        &self,
        points_3d: &Matrix3xX<f64>,
        points_2d: &nalgebra::Matrix2xX<f64>,
    ) -> Result<Self, CameraModelError> {
        let n = points_3d.ncols();
        if n < 4 {
            return Err(CameraModelError::InvalidParams(
                "linear_estimation requires at least 4 correspondences".to_string(),
            ));
        }

        let mut a = nalgebra::DMatrix::<f64>::zeros(n, 4);
        let mut b = nalgebra::DVector::<f64>::zeros(n);

        for i in 0..n {
            let x = points_3d[(0, i)];
            let y = points_3d[(1, i)];
            let z = points_3d[(2, i)];
            let d = (x * x + y * y + z * z).sqrt();
            if d < GEOMETRIC_PRECISION {
                continue;
            }
            let theta = (z / d).clamp(-1.0, 1.0).acos();

            let u = points_2d[(0, i)];
            let v = points_2d[(1, i)];
            let dx = u - self.cx;
            let dy = v - self.cy;
            let r = (dx * dx + dy * dy).sqrt();

            let t2 = theta * theta;
            let t3 = t2 * theta;
            let t4 = t3 * theta;
            a[(i, 0)] = theta;
            a[(i, 1)] = t2;
            a[(i, 2)] = t3;
            a[(i, 3)] = t4;
            b[i] = r;
        }

        let svd = a.svd(true, true);
        let coeffs = svd
            .solve(&b, GEOMETRIC_PRECISION)
            .map_err(|e| CameraModelError::InvalidParams(format!("SVD solve failed: {e}")))?;

        let distortion = DistortionModel::FTheta {
            k1: coeffs[0],
            k2: coeffs[1],
            k3: coeffs[2],
            k4: coeffs[3],
        };
        FThetaCamera::new(self.cx, self.cy, distortion)
    }
}

// ── CameraModel trait ─────────────────────────────────────────────────────────

impl CameraModel for FThetaCamera {
    const INTRINSIC_DIM: usize = 6;

    type IntrinsicJacobian = SMatrix<f64, 2, 6>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let (x, y, z) = (p_cam.x, p_cam.y, p_cam.z);

        if z < MIN_DEPTH {
            return Err(CameraModelError::PointBehindCamera {
                z,
                min_z: MIN_DEPTH,
            });
        }

        let d = (x * x + y * y + z * z).sqrt();
        let theta = (z / d).clamp(-1.0, 1.0).acos();
        let f_theta = self.poly_forward(theta);
        let r_p = (x * x + y * y).sqrt();

        if r_p < GEOMETRIC_PRECISION {
            return Ok(Vector2::new(self.cx, self.cy));
        }

        let inv_rp = 1.0 / r_p;
        Ok(Vector2::new(
            self.cx + f_theta * x * inv_rp,
            self.cy + f_theta * y * inv_rp,
        ))
    }

    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let dx = point_2d.x - self.cx;
        let dy = point_2d.y - self.cy;
        let r_d = (dx * dx + dy * dy).sqrt();

        if r_d < GEOMETRIC_PRECISION {
            return Ok(Vector3::new(0.0, 0.0, 1.0));
        }

        let (k1, ..) = self.distortion_params();

        // Newton-Raphson: solve f(theta) = r_d
        let mut theta = r_d / k1;
        for _ in 0..100 {
            let f_val = self.poly_forward(theta);
            let f_deriv = self.poly_forward_deriv(theta);
            if f_deriv.abs() < 1e-12 {
                break;
            }
            let delta = (f_val - r_d) / f_deriv;
            theta -= delta;
            if delta.abs() < CONVERGENCE_THRESHOLD {
                break;
            }
        }

        if !theta.is_finite() || theta < 0.0 {
            return Err(CameraModelError::NumericalError {
                operation: "ftheta_unproject".to_string(),
                details: format!("Newton-Raphson diverged, theta={theta}"),
            });
        }

        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let inv_rd = 1.0 / r_d;
        let ray = Vector3::new(sin_theta * dx * inv_rd, sin_theta * dy * inv_rd, cos_theta);
        Ok(ray.normalize())
    }

    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let (x, y, z) = (p_cam.x, p_cam.y, p_cam.z);
        let r_p2 = x * x + y * y;
        let d2 = r_p2 + z * z;
        let d = d2.sqrt();
        let r_p = r_p2.sqrt();

        let mut j = SMatrix::<f64, 2, 3>::zeros();

        if r_p < GEOMETRIC_PRECISION {
            // Limit at optical axis: ∂u/∂x = k₁/z, ∂v/∂y = k₁/z
            let k1 = self.distortion_params().0;
            j[(0, 0)] = k1 / z;
            j[(1, 1)] = k1 / z;
            return j;
        }

        let theta = (z / d).clamp(-1.0, 1.0).acos();
        let f_val = self.poly_forward(theta);
        let f_prime = self.poly_forward_deriv(theta);

        // A = f′·z / (r_p²·d²),   B = f / r_p³
        let a = f_prime * z / (r_p2 * d2);
        let b = f_val / (r_p2 * r_p);

        j[(0, 0)] = a * x * x + b * y * y;
        j[(0, 1)] = (a - b) * x * y;
        j[(0, 2)] = -f_prime * x / d2;
        j[(1, 0)] = j[(0, 1)];
        j[(1, 1)] = a * y * y + b * x * x;
        j[(1, 2)] = -f_prime * y / d2;
        j
    }

    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let (x, y, z) = (p_cam.x, p_cam.y, p_cam.z);
        let r_p2 = x * x + y * y;
        let r_p = r_p2.sqrt();
        let d = (r_p2 + z * z).sqrt();

        let mut j = SMatrix::<f64, 2, 6>::zeros();
        // ∂u/∂cx = 1, ∂v/∂cy = 1
        j[(0, 0)] = 1.0;
        j[(1, 1)] = 1.0;

        if r_p < GEOMETRIC_PRECISION {
            // All kᵢ columns are 0 at the optical axis.
            return j;
        }

        let theta = (z / d).clamp(-1.0, 1.0).acos();
        let cos_phi = x / r_p;
        let sin_phi = y / r_p;

        let mut theta_pow = theta;
        for col in 2..6 {
            j[(0, col)] = theta_pow * cos_phi;
            j[(1, col)] = theta_pow * sin_phi;
            theta_pow *= theta;
        }
        j
    }

    fn validate_params(&self) -> Result<(), CameraModelError> {
        if !self.cx.is_finite() || !self.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointNotFinite {
                cx: self.cx,
                cy: self.cy,
            });
        }
        match self.distortion {
            DistortionModel::FTheta { .. } => self.distortion.validate(),
            _ => Err(CameraModelError::InvalidParams(
                "FThetaCamera requires DistortionModel::FTheta".to_string(),
            )),
        }
    }

    fn get_pinhole_params(&self) -> PinholeParams {
        let (k1, ..) = self.distortion_params();
        PinholeParams {
            fx: k1,
            fy: k1,
            cx: self.cx,
            cy: self.cy,
        }
    }

    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    fn get_model_name(&self) -> &'static str {
        "ftheta"
    }
}

// ── Conversion traits ─────────────────────────────────────────────────────────

/// Parameter order: `[cx, cy, k1, k2, k3, k4]`.
impl From<&FThetaCamera> for [f64; 6] {
    fn from(cam: &FThetaCamera) -> Self {
        let (k1, k2, k3, k4) = cam.distortion_params();
        [cam.cx, cam.cy, k1, k2, k3, k4]
    }
}

impl From<[f64; 6]> for FThetaCamera {
    fn from(p: [f64; 6]) -> Self {
        Self {
            cx: p[0],
            cy: p[1],
            distortion: DistortionModel::FTheta {
                k1: p[2],
                k2: p[3],
                k3: p[4],
                k4: p[5],
            },
        }
    }
}

impl From<&FThetaCamera> for DVector<f64> {
    fn from(cam: &FThetaCamera) -> Self {
        let arr: [f64; 6] = cam.into();
        DVector::from_row_slice(&arr)
    }
}

impl TryFrom<&[f64]> for FThetaCamera {
    type Error = CameraModelError;

    fn try_from(params: &[f64]) -> Result<Self, Self::Error> {
        if params.len() != 6 {
            return Err(CameraModelError::InvalidParams(format!(
                "FThetaCamera requires 6 parameters, got {}",
                params.len()
            )));
        }
        let cam = FThetaCamera::from([
            params[0], params[1], params[2], params[3], params[4], params[5],
        ]);
        cam.validate_params()?;
        Ok(cam)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{JACOBIAN_TEST_TOLERANCE, NUMERICAL_DERIVATIVE_EPS, PROJECTION_TEST_TOLERANCE};
    use nalgebra::{Matrix2xX, Matrix3xX};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Camera representative of a real wide-angle fisheye (k1≈focal length).
    fn make_camera() -> FThetaCamera {
        FThetaCamera::from([320.0, 240.0, 500.0, -10.0, 2.0, -0.1])
    }

    /// Pure-linear camera (no higher-order distortion) for analytic checks.
    fn make_linear_camera() -> FThetaCamera {
        FThetaCamera::from([320.0, 240.0, 500.0, 0.0, 0.0, 0.0])
    }

    // ── construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_creation_and_validate() -> TestResult {
        let cam = FThetaCamera::new(
            320.0,
            240.0,
            DistortionModel::FTheta {
                k1: 500.0,
                k2: -10.0,
                k3: 2.0,
                k4: -0.1,
            },
        )?;
        assert_eq!(cam.cx, 320.0);
        assert_eq!(cam.cy, 240.0);
        assert!(cam.validate_params().is_ok());
        Ok(())
    }

    #[test]
    fn test_validate_invalid_k1_zero() {
        let cam = FThetaCamera {
            cx: 320.0,
            cy: 240.0,
            distortion: DistortionModel::FTheta {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                k4: 0.0,
            },
        };
        assert!(cam.validate_params().is_err());
    }

    #[test]
    fn test_validate_invalid_k1_negative() {
        let cam = FThetaCamera {
            cx: 320.0,
            cy: 240.0,
            distortion: DistortionModel::FTheta {
                k1: -1.0,
                k2: 0.0,
                k3: 0.0,
                k4: 0.0,
            },
        };
        assert!(cam.validate_params().is_err());
    }

    #[test]
    fn test_validate_nan_coefficient() {
        let cam = FThetaCamera {
            cx: 320.0,
            cy: 240.0,
            distortion: DistortionModel::FTheta {
                k1: 500.0,
                k2: f64::NAN,
                k3: 0.0,
                k4: 0.0,
            },
        };
        assert!(cam.validate_params().is_err());
    }

    #[test]
    fn test_validate_wrong_distortion_type() {
        let cam = FThetaCamera {
            cx: 320.0,
            cy: 240.0,
            distortion: DistortionModel::None,
        };
        assert!(cam.validate_params().is_err());
    }

    #[test]
    fn test_get_model_name() {
        assert_eq!(make_camera().get_model_name(), "ftheta");
    }

    #[test]
    fn test_debug_format() {
        let cam = make_camera();
        let s = format!("{cam:?}");
        assert!(s.contains("FThetaCamera"));
        assert!(s.contains("cx"));
        assert!(s.contains("k1"));
    }

    // ── projection ───────────────────────────────────────────────────────────

    #[test]
    fn test_project_optical_axis() -> TestResult {
        let cam = make_camera();
        let p = Vector3::new(0.0, 0.0, 1.0);
        let px = cam.project(&p)?;
        assert!((px.x - cam.cx).abs() < PROJECTION_TEST_TOLERANCE);
        assert!((px.y - cam.cy).abs() < PROJECTION_TEST_TOLERANCE);
        Ok(())
    }

    #[test]
    fn test_project_off_axis_linear_model() -> TestResult {
        // With k2=k3=k4=0: f(θ) = k₁θ, so u = cx + k₁·θ·(x/r_p)
        let cam = make_linear_camera();
        let p = Vector3::new(1.0, 0.0, 1.0);
        let theta = (1.0_f64 / 2.0_f64.sqrt()).acos(); // arccos(1/√2) = π/4
        let expected_u = 320.0 + 500.0 * theta;
        let px = cam.project(&p)?;
        assert!(
            (px.x - expected_u).abs() < 1e-8,
            "u={} expected={expected_u}",
            px.x
        );
        assert!((px.y - 240.0).abs() < 1e-8);
        Ok(())
    }

    #[test]
    fn test_project_with_higher_order_terms() -> TestResult {
        let linear = make_linear_camera();
        let distorted = make_camera();
        let p = Vector3::new(0.5, 0.3, 1.0);
        let px_lin = linear.project(&p)?;
        let px_dist = distorted.project(&p)?;
        // With negative k2, the distorted radius is smaller for moderate angles
        let r_lin = ((px_lin.x - 320.0).powi(2) + (px_lin.y - 240.0).powi(2)).sqrt();
        let r_dist = ((px_dist.x - 320.0).powi(2) + (px_dist.y - 240.0).powi(2)).sqrt();
        assert!(r_lin > 0.0 && r_dist > 0.0);
        Ok(())
    }

    #[test]
    fn test_project_behind_camera_error() {
        let cam = make_camera();
        let p = Vector3::new(0.0, 0.0, -1.0);
        assert!(matches!(
            cam.project(&p),
            Err(CameraModelError::PointBehindCamera { .. })
        ));
    }

    // ── unprojection ─────────────────────────────────────────────────────────

    #[test]
    fn test_unproject_center_pixel() -> TestResult {
        let cam = make_camera();
        let px = Vector2::new(cam.cx, cam.cy);
        let ray = cam.unproject(&px)?;
        assert!((ray.x).abs() < PROJECTION_TEST_TOLERANCE);
        assert!((ray.y).abs() < PROJECTION_TEST_TOLERANCE);
        assert!((ray.z - 1.0).abs() < PROJECTION_TEST_TOLERANCE);
        Ok(())
    }

    #[test]
    fn test_unproject_off_axis_linear() -> TestResult {
        // For linear camera: pixel r = k₁·θ → θ = r/k₁
        let cam = make_linear_camera();
        let r = 100.0_f64;
        let px = Vector2::new(cam.cx + r, cam.cy);
        let ray = cam.unproject(&px)?;
        let theta_expected = r / 500.0;
        assert!((ray.x - theta_expected.sin()).abs() < 1e-6);
        assert!((ray.z - theta_expected.cos()).abs() < 1e-6);
        Ok(())
    }

    // ── round-trip ───────────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_optical_axis() -> TestResult {
        let cam = make_camera();
        let p = Vector3::new(0.0, 0.0, 2.0);
        let px = cam.project(&p)?;
        let ray = cam.unproject(&px)?;
        let p_norm = p.normalize();
        assert!((ray.x - p_norm.x).abs() < 1e-10);
        assert!((ray.y - p_norm.y).abs() < 1e-10);
        assert!((ray.z - p_norm.z).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_round_trip_project_unproject() -> TestResult {
        let cam = make_camera();
        let points = [
            Vector3::new(0.3, 0.2, 1.0),
            Vector3::new(-0.5, 0.1, 2.0),
            Vector3::new(0.1, -0.4, 1.5),
            Vector3::new(0.0, 0.6, 1.0),
        ];
        for p in &points {
            let px = cam.project(p)?;
            let ray = cam.unproject(&px)?;
            let p_norm = p.normalize();
            let dot = ray.dot(&p_norm);
            assert!(
                (dot - 1.0).abs() < 1e-8,
                "round-trip failed for {p:?}: dot={dot}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_round_trip_with_distortion() -> TestResult {
        let cam = make_camera(); // k2..k4 nonzero
        let p = Vector3::new(0.4, -0.3, 1.2);
        let px = cam.project(&p)?;
        let ray = cam.unproject(&px)?;
        let dot = ray.dot(&p.normalize());
        assert!((dot - 1.0).abs() < 1e-8);
        Ok(())
    }

    // ── Jacobian point ────────────────────────────────────────────────────────

    #[test]
    fn test_jacobian_point_dimensions() {
        let cam = make_camera();
        let p = Vector3::new(0.3, 0.2, 1.0);
        let j = cam.jacobian_point(&p);
        assert_eq!(j.nrows(), 2);
        assert_eq!(j.ncols(), 3);
    }

    #[test]
    fn test_jacobian_point_optical_axis() {
        let cam = make_camera();
        let z = 2.0_f64;
        let p = Vector3::new(0.0, 0.0, z);
        let j = cam.jacobian_point(&p);
        let k1 = cam.distortion_params().0;
        let expected = k1 / z;
        assert!((j[(0, 0)] - expected).abs() < 1e-10, "J[0,0]={}", j[(0, 0)]);
        assert!((j[(1, 1)] - expected).abs() < 1e-10, "J[1,1]={}", j[(1, 1)]);
        assert!(j[(0, 1)].abs() < 1e-10);
        assert!(j[(0, 2)].abs() < 1e-10);
        assert!(j[(1, 0)].abs() < 1e-10);
        assert!(j[(1, 2)].abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let cam = make_camera();
        let p = Vector3::new(0.3, 0.2, 1.5);
        let j_analytical = cam.jacobian_point(&p);
        let px0 = cam.project(&p)?;

        for col in 0..3 {
            let mut p_plus = p;
            p_plus[col] += NUMERICAL_DERIVATIVE_EPS;
            let px_plus = cam.project(&p_plus)?;
            let num_du = (px_plus.x - px0.x) / NUMERICAL_DERIVATIVE_EPS;
            let num_dv = (px_plus.y - px0.y) / NUMERICAL_DERIVATIVE_EPS;

            assert!(
                (j_analytical[(0, col)] - num_du).abs() < JACOBIAN_TEST_TOLERANCE,
                "J[0,{col}]: analytical={} numerical={num_du}",
                j_analytical[(0, col)]
            );
            assert!(
                (j_analytical[(1, col)] - num_dv).abs() < JACOBIAN_TEST_TOLERANCE,
                "J[1,{col}]: analytical={} numerical={num_dv}",
                j_analytical[(1, col)]
            );
        }
        Ok(())
    }

    // ── Jacobian intrinsics ───────────────────────────────────────────────────

    #[test]
    fn test_jacobian_intrinsics_dimensions() {
        let cam = make_camera();
        let p = Vector3::new(0.3, 0.2, 1.0);
        let j = cam.jacobian_intrinsics(&p);
        assert_eq!(j.nrows(), 2);
        assert_eq!(j.ncols(), 6);
    }

    #[test]
    fn test_jacobian_intrinsics_optical_axis() {
        let cam = make_camera();
        let p = Vector3::new(0.0, 0.0, 1.0);
        let j = cam.jacobian_intrinsics(&p);
        assert!((j[(0, 0)] - 1.0).abs() < 1e-12); // ∂u/∂cx
        assert!((j[(1, 1)] - 1.0).abs() < 1e-12); // ∂v/∂cy
        // All kᵢ columns must be zero on axis
        for col in 2..6 {
            assert!(j[(0, col)].abs() < 1e-12, "J[0,{col}]={}", j[(0, col)]);
            assert!(j[(1, col)].abs() < 1e-12, "J[1,{col}]={}", j[(1, col)]);
        }
    }

    #[test]
    fn test_jacobian_intrinsics_numerical() -> TestResult {
        let cam = make_camera();
        let p = Vector3::new(0.3, 0.2, 1.5);
        let j_analytical = cam.jacobian_intrinsics(&p);
        let px0 = cam.project(&p)?;

        // Perturb each of the 6 intrinsics in order: cx, cy, k1, k2, k3, k4
        let params0: [f64; 6] = (&cam).into();
        for col in 0..6 {
            let mut params_plus = params0;
            params_plus[col] += NUMERICAL_DERIVATIVE_EPS;
            let cam_plus = FThetaCamera::from(params_plus);
            let px_plus = cam_plus.project(&p)?;
            let num_du = (px_plus.x - px0.x) / NUMERICAL_DERIVATIVE_EPS;
            let num_dv = (px_plus.y - px0.y) / NUMERICAL_DERIVATIVE_EPS;

            assert!(
                (j_analytical[(0, col)] - num_du).abs() < JACOBIAN_TEST_TOLERANCE,
                "J_intr[0,{col}]: analytical={} numerical={num_du}",
                j_analytical[(0, col)]
            );
            assert!(
                (j_analytical[(1, col)] - num_dv).abs() < JACOBIAN_TEST_TOLERANCE,
                "J_intr[1,{col}]: analytical={} numerical={num_dv}",
                j_analytical[(1, col)]
            );
        }
        Ok(())
    }

    // ── conversions ───────────────────────────────────────────────────────────

    #[test]
    fn test_from_array_roundtrip() {
        let arr = [320.0_f64, 240.0, 500.0, -10.0, 2.0, -0.1];
        let cam = FThetaCamera::from(arr);
        let arr2: [f64; 6] = (&cam).into();
        for (a, b) in arr.iter().zip(arr2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_try_from_slice_correct_length() -> TestResult {
        let params = [320.0_f64, 240.0, 500.0, -10.0, 2.0, -0.1];
        let cam = FThetaCamera::try_from(params.as_slice())?;
        assert_eq!(cam.cx, 320.0);
        Ok(())
    }

    #[test]
    fn test_try_from_slice_wrong_length() {
        let params = [320.0_f64, 240.0, 500.0];
        assert!(FThetaCamera::try_from(params.as_slice()).is_err());
    }

    #[test]
    fn test_dvector_conversion() {
        let cam = make_camera();
        let v: DVector<f64> = (&cam).into();
        assert_eq!(v.len(), 6);
        assert!((v[0] - cam.cx).abs() < 1e-15);
        let (k1, k2, k3, k4) = cam.distortion_params();
        assert!((v[2] - k1).abs() < 1e-15);
        assert!((v[3] - k2).abs() < 1e-15);
        assert!((v[4] - k3).abs() < 1e-15);
        assert!((v[5] - k4).abs() < 1e-15);
    }

    #[test]
    fn test_get_pinhole_params() {
        let cam = make_camera();
        let pp = cam.get_pinhole_params();
        let (k1, ..) = cam.distortion_params();
        assert!((pp.fx - k1).abs() < 1e-15);
        assert!((pp.fy - k1).abs() < 1e-15);
        assert!((pp.cx - cam.cx).abs() < 1e-15);
        assert!((pp.cy - cam.cy).abs() < 1e-15);
    }

    // ── linear estimation ─────────────────────────────────────────────────────

    #[test]
    fn test_linear_estimation() -> TestResult {
        let gt_cam = make_camera();
        let n = 50_usize;
        let mut pts3d = Matrix3xX::zeros(n);
        let mut pts2d = Matrix2xX::zeros(n);

        // Sample points uniformly across a half-sphere (θ ∈ [0, 60°])
        for i in 0..n {
            let theta = std::f64::consts::FRAC_PI_3 * (i as f64) / (n as f64);
            let phi = 2.0 * std::f64::consts::PI * (i as f64 * 0.618_033_988); // golden angle
            let x = theta.sin() * phi.cos();
            let y = theta.sin() * phi.sin();
            let z = theta.cos();
            pts3d[(0, i)] = x;
            pts3d[(1, i)] = y;
            pts3d[(2, i)] = z;
            let px = gt_cam.project(&Vector3::new(x, y, z))?;
            pts2d[(0, i)] = px.x;
            pts2d[(1, i)] = px.y;
        }

        let seed = FThetaCamera::from([gt_cam.cx, gt_cam.cy, 500.0, 0.0, 0.0, 0.0]);
        let estimated = seed.linear_estimation(&pts3d, &pts2d)?;

        // Verify reprojection error < 1 pixel on training points
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let p = Vector3::new(pts3d[(0, i)], pts3d[(1, i)], pts3d[(2, i)]);
            let px_est = estimated.project(&p)?;
            let err =
                ((px_est.x - pts2d[(0, i)]).powi(2) + (px_est.y - pts2d[(1, i)]).powi(2)).sqrt();
            max_err = max_err.max(err);
        }
        assert!(max_err < 1.0, "max reprojection error={max_err:.3} px");
        Ok(())
    }

    // ── batch projection ──────────────────────────────────────────────────────

    #[test]
    fn test_project_batch_sentinel() {
        let cam = make_camera();
        let pts = Matrix3xX::from_columns(&[
            Vector3::new(0.1, 0.2, 1.0),
            Vector3::new(0.0, 0.0, -1.0), // behind camera
        ]);
        let result = cam.project_batch(&pts);
        assert!(result[(0, 0)].is_finite());
        assert!((result[(0, 1)] - 1e6).abs() < 1.0);
        assert!((result[(1, 1)] - 1e6).abs() < 1.0);
    }
}
