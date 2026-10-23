//! The mathematics shared by [`Fisheye62Camera`](crate::Fisheye62Camera) and
//! [`Fisheye624Camera`](crate::Fisheye624Camera).
//!
//! Fisheye62 is Fisheye624 with the thin-prism terms `s0..s3` zero, so both
//! types delegate here and cannot drift apart. The model is Project Aria's
//! `FisheyeRadTanThinPrism` (`numK = 6`); see the module documentation of
//! [`crate::fisheye624`] for the formulas and the source.
//!
//! Parameter order of the 16 intrinsics, which is also the order of the
//! columns of [`Core::jacobian_params`]:
//! `fx fy cx cy k0 k1 k2 k3 k4 k5 p0 p1 s0 s1 s2 s3`.

use crate::{CameraModelError, GEOMETRIC_PRECISION};
use nalgebra::{Matrix2, SMatrix, Vector2, Vector3};

/// Radial terms `k0..k5`.
pub(crate) const NUM_K: usize = 6;

/// Newton iterations for the two inversions in `unproject`.
const MAX_ITERATIONS: usize = 50;

/// A Newton step below this ends an inversion.
const STEP_TOLERANCE: f64 = 1e-14;

/// An inversion whose forward residual is above this did not converge.
const RESIDUAL_TOLERANCE: f64 = 1e-9;

/// Largest incidence angle `unproject` returns: `project` needs
/// `z >= GEOMETRIC_PRECISION`, i.e. θ just below π/2.
const MAX_THETA: f64 = std::f64::consts::FRAC_PI_2 - GEOMETRIC_PRECISION;

/// All sixteen intrinsics of the model, by value.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Core {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k: [f64; NUM_K],
    pub p: [f64; 2],
    pub s: [f64; 4],
}

impl Core {
    /// `R(θ) = θ + k0 θ³ + k1 θ⁵ + … + k5 θ¹³` and `dR/dθ`.
    fn radial(&self, theta: f64) -> (f64, f64) {
        let t = theta * theta;
        let mut power = t;
        let mut sum = 0.0;
        let mut derivative = 1.0;
        for (i, k) in self.k.iter().enumerate() {
            sum += k * power;
            derivative += (2 * i + 3) as f64 * k * power;
            power *= t;
        }
        (theta * (1.0 + sum), derivative)
    }

    /// The tangential and thin-prism offsets at the radially distorted point
    /// `(xr, yr)`, and their 2×2 Jacobian with respect to it.
    fn offsets(&self, xr: f64, yr: f64) -> (Vector2<f64>, Matrix2<f64>) {
        let [p0, p1] = self.p;
        let [s0, s1, s2, s3] = self.s;
        let rd2 = xr * xr + yr * yr;
        let rd4 = rd2 * rd2;
        let tx = p0 * (2.0 * xr * xr + rd2) + 2.0 * p1 * xr * yr;
        let ty = p1 * (2.0 * yr * yr + rd2) + 2.0 * p0 * xr * yr;
        let offset = Vector2::new(tx + s0 * rd2 + s1 * rd4, ty + s2 * rd2 + s3 * rd4);

        let tpx = s0 + 2.0 * s1 * rd2;
        let tpy = s2 + 2.0 * s3 * rd2;
        let jacobian = Matrix2::new(
            6.0 * p0 * xr + 2.0 * p1 * yr + 2.0 * xr * tpx,
            2.0 * p0 * yr + 2.0 * p1 * xr + 2.0 * yr * tpx,
            2.0 * p1 * xr + 2.0 * p0 * yr + 2.0 * xr * tpy,
            6.0 * p1 * yr + 2.0 * p0 * xr + 2.0 * yr * tpy,
        );
        (offset, jacobian)
    }

    /// `(xr, yr)`, the point after the radial polynomial, and `(θ, ρ)` with
    /// `ρ = √(x² + y²)`. Requires `z > 0`.
    fn radially_distorted(&self, p: &Vector3<f64>) -> (Vector2<f64>, f64, f64) {
        let rho = p.x.hypot(p.y);
        let theta = rho.atan2(p.z);
        let (radius, _) = self.radial(theta);
        // R/ρ tends to 1/z on the optical axis.
        let scale = if rho < GEOMETRIC_PRECISION {
            1.0 / p.z
        } else {
            radius / rho
        };
        (Vector2::new(scale * p.x, scale * p.y), theta, rho)
    }

    pub(crate) fn project(&self, p: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        if p.z < GEOMETRIC_PRECISION {
            return Err(CameraModelError::PointBehindCamera {
                z: p.z,
                min_z: GEOMETRIC_PRECISION,
            });
        }
        let (xr, _, _) = self.radially_distorted(p);
        let (offset, _) = self.offsets(xr.x, xr.y);
        Ok(Vector2::new(
            self.fx * (xr.x + offset.x) + self.cx,
            self.fy * (xr.y + offset.y) + self.cy,
        ))
    }

    /// Solves `R(θ) = target` for θ in `[0, MAX_THETA]`.
    fn theta_from_radius(&self, target: f64) -> Result<f64, CameraModelError> {
        let fail = |details: String| CameraModelError::NumericalError {
            operation: "unprojection".to_string(),
            details,
        };

        // Newton from θ = target, kept inside the half space.
        let mut theta = target.min(MAX_THETA);
        for _ in 0..MAX_ITERATIONS {
            let (radius, derivative) = self.radial(theta);
            if derivative.abs() < f64::EPSILON {
                break;
            }
            let step = (radius - target) / derivative;
            theta -= step;
            if !(0.0..=MAX_THETA).contains(&theta) {
                theta = theta.clamp(0.0, MAX_THETA);
            }
            if step.abs() < STEP_TOLERANCE {
                break;
            }
        }
        if (self.radial(theta).0 - target).abs() < RESIDUAL_TOLERANCE {
            return Ok(theta);
        }

        // Bisection, when the polynomial rises through `target` once.
        let (mut low, mut high) = (0.0, MAX_THETA);
        if self.radial(high).0 < target {
            return Err(fail(format!(
                "radius {target} lies beyond the model's field of view"
            )));
        }
        for _ in 0..200 {
            let middle = 0.5 * (low + high);
            if self.radial(middle).0 < target {
                low = middle;
            } else {
                high = middle;
            }
        }
        let theta = 0.5 * (low + high);
        if (self.radial(theta).0 - target).abs() < RESIDUAL_TOLERANCE {
            Ok(theta)
        } else {
            Err(fail(format!("no incidence angle has radius {target}")))
        }
    }

    pub(crate) fn unproject(&self, uv: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let fail = |details: String| CameraModelError::NumericalError {
            operation: "unprojection".to_string(),
            details,
        };
        let target = Vector2::new((uv.x - self.cx) / self.fx, (uv.y - self.cy) / self.fy);
        if !target.x.is_finite() || !target.y.is_finite() {
            return Err(fail("pixel is not finite".to_string()));
        }

        // Undo the tangential and thin-prism offsets: solve `ξ + o(ξ) = target`.
        let mut xi = target;
        let mut converged = false;
        for _ in 0..MAX_ITERATIONS {
            let (offset, jacobian) = self.offsets(xi.x, xi.y);
            let full = Matrix2::identity() + jacobian;
            let Some(inverse) = full.try_inverse() else {
                return Err(fail("singular distortion Jacobian".to_string()));
            };
            let step = inverse * (target - xi - offset);
            xi += step;
            if step.norm() < STEP_TOLERANCE {
                converged = true;
                break;
            }
        }
        let (offset, _) = self.offsets(xi.x, xi.y);
        if !converged && (xi + offset - target).norm() > RESIDUAL_TOLERANCE {
            return Err(fail("tangential inversion did not converge".to_string()));
        }

        let radius = xi.norm();
        if radius == 0.0 {
            return Ok(Vector3::new(0.0, 0.0, 1.0));
        }
        let theta = self.theta_from_radius(radius)?;
        let (sin, cos) = theta.sin_cos();
        Ok(Vector3::new(sin * xi.x / radius, sin * xi.y / radius, cos))
    }

    /// ∂ξ/∂(x, y, z) for `ξ = (R(θ)/ρ)·(x, y)`.
    fn radial_point_jacobian(&self, p: &Vector3<f64>) -> SMatrix<f64, 2, 3> {
        let (x, y, z) = (p.x, p.y, p.z);
        let rho = x.hypot(y);
        if rho < GEOMETRIC_PRECISION {
            let inv_z = 1.0 / z;
            return SMatrix::<f64, 2, 3>::new(
                inv_z,
                0.0,
                -x * inv_z * inv_z,
                0.0,
                inv_z,
                -y * inv_z * inv_z,
            );
        }
        let theta = rho.atan2(z);
        let (radius, derivative) = self.radial(theta);
        let norm_sq = rho * rho + z * z;

        // θ and ρ as functions of (x, y, z).
        let theta_grad = Vector3::new(
            z * x / (rho * norm_sq),
            z * y / (rho * norm_sq),
            -rho / norm_sq,
        );
        let rho_grad = Vector3::new(x / rho, y / rho, 0.0);

        // g = R(θ)/ρ.
        let g = radius / rho;
        let g_grad = theta_grad * (derivative / rho) - rho_grad * (radius / (rho * rho));

        let mut jacobian = SMatrix::<f64, 2, 3>::zeros();
        for column in 0..3 {
            let identity_x = if column == 0 { g } else { 0.0 };
            let identity_y = if column == 1 { g } else { 0.0 };
            jacobian[(0, column)] = identity_x + x * g_grad[column];
            jacobian[(1, column)] = identity_y + y * g_grad[column];
        }
        jacobian
    }

    pub(crate) fn jacobian_point(&self, p: &Vector3<f64>) -> SMatrix<f64, 2, 3> {
        let (xr, _, _) = self.radially_distorted(p);
        let (_, offsets_jacobian) = self.offsets(xr.x, xr.y);
        let full = Matrix2::identity() + offsets_jacobian;
        let d_xi = self.radial_point_jacobian(p);
        let scaled = full * d_xi;
        SMatrix::<f64, 2, 3>::from_fn(|row, column| {
            let focal = if row == 0 { self.fx } else { self.fy };
            focal * scaled[(row, column)]
        })
    }

    /// ∂(u, v)/∂ `[fx fy cx cy k0..k5 p0 p1 s0..s3]`.
    pub(crate) fn jacobian_params(&self, p: &Vector3<f64>) -> SMatrix<f64, 2, 16> {
        let mut jacobian = SMatrix::<f64, 2, 16>::zeros();
        jacobian[(0, 2)] = 1.0;
        jacobian[(1, 3)] = 1.0;

        let (xr, theta, rho) = self.radially_distorted(p);
        let (offset, offsets_jacobian) = self.offsets(xr.x, xr.y);
        let (xr_x, xr_y) = (xr.x, xr.y);
        let rd2 = xr_x * xr_x + xr_y * xr_y;
        let rd4 = rd2 * rd2;

        // The distorted point, and ∂(u, v)/∂(fx, fy).
        jacobian[(0, 0)] = xr_x + offset.x;
        jacobian[(1, 1)] = xr_y + offset.y;

        // Radial terms: ∂ξ/∂k_i = θ^(2i+3)·(x, y)/ρ, mapped through I + ∂o/∂ξ.
        let direction = if rho < GEOMETRIC_PRECISION {
            Vector2::zeros()
        } else {
            Vector2::new(p.x / rho, p.y / rho)
        };
        let full = Matrix2::identity() + offsets_jacobian;
        let mut power = theta * theta * theta;
        for i in 0..NUM_K {
            let d = full * (direction * power);
            jacobian[(0, 4 + i)] = self.fx * d.x;
            jacobian[(1, 4 + i)] = self.fy * d.y;
            power *= theta * theta;
        }

        // Tangential terms.
        jacobian[(0, 10)] = self.fx * (2.0 * xr_x * xr_x + rd2);
        jacobian[(1, 10)] = self.fy * (2.0 * xr_x * xr_y);
        jacobian[(0, 11)] = self.fx * (2.0 * xr_x * xr_y);
        jacobian[(1, 11)] = self.fy * (2.0 * xr_y * xr_y + rd2);

        // Thin-prism terms.
        jacobian[(0, 12)] = self.fx * rd2;
        jacobian[(0, 13)] = self.fx * rd4;
        jacobian[(1, 14)] = self.fy * rd2;
        jacobian[(1, 15)] = self.fy * rd4;
        jacobian
    }
}
