//! Direct photometric residual of an inverse-range point between two `SE_2(3)`
//! states: the residual of DSO (Engel, Koltun, Cremers, TPAMI 2018) and
//! DM-VIO (von Stumberg, Cremers, RA-L 2022) as a factor of this graph.
//!
//! A point lives in its **host** frame at a pixel `q₀` with an inverse range
//! `ρ` along the unit bearing `b₀` of that pixel, the same variable the
//! geometric inverse-range factor uses. It carries the host intensities `Iₕ(qₖ)`
//! and gradient weights `wₖ` of an 8-pixel pattern `qₖ = q₀ + δₖ`, fixed when
//! the point was made. In a **target** frame the same pixels must look alike,
//! up to a brightness transfer:
//!
//! ```text
//! z₀     = b₀.z / ρ                          depth of the centre pixel in the host camera
//! Pₕ,ₖ   = z₀ · K⁻¹ q̃ₖ                       the pattern shares the centre's depth
//! P_t,ₖ  = T_SC⁻¹ X_t⁻¹ X_h T_SC Pₕ,ₖ         target camera
//! q'ₖ    = π(P_t,ₖ)
//! α      = (t_t / t_h) · e^{a_t − a_h}
//! rₖ     = sₖ · ( I_t[q'ₖ] − b_t − α · (Iₕ(qₖ) − b_h) )
//! sₖ     = σ⁻¹ · √W · √hwₖ · wₖ              hwₖ = min(1, γ / |raw rₖ|)
//! ```
//!
//! `(a, b)` are the per-frame affine brightness parameters, `t` the exposures,
//! `γ` the Huber threshold, `W` DM-VIO's dynamic photometric weight and `σ` a
//! scale in grey levels. The Huber weight enters as iteratively reweighted
//! least squares: it is held fixed in the Jacobian.
//!
//! # Jacobians
//!
//! Analytic, right perturbation on `SE_2(3)` and `SE_3` (tangent `[ρ, θ, ν]`
//! and `[ρ, θ]`). With `g = ∇I_t(q')ᵀ Jπ` the image gradient pulled back to the
//! target camera, `u = T_SC Pₕ`, `w = X_h u`, `v = X_t⁻¹ w`, `P_t = T_SC⁻¹ v`,
//! `M = R_scᵀ R_tᵀ`, `R_rel = M R_h R_sc`, and `s` the row scale:
//!
//! ```text
//! host   ρ: s·g M R_h          θ: −s·g M R_h [u]×          ν: 0
//! target ρ: −s·g R_scᵀ         θ:  s·g R_scᵀ [v]×          ν: 0
//! T_SC   ρ: s·g (R_rel − I)    θ:  s·g ( [P_t]× − R_rel [Pₕ]× )
//! ρ_inv:    −s·g R_rel Pₕ / ρ
//! affine:   ∂/∂b_t = −s, ∂/∂b_h = s·α, ∂/∂a_t = −s·α·(Iₕ−b_h), ∂/∂a_h = s·α·(Iₕ−b_h)
//! ```
//!
//! # Scale
//!
//! The states live in a visual frame whose metric scale is `s` (see
//! [`ScaleGravityImuFactor`](crate::factors::imu::se23::ScaleGravityImuFactor)),
//! so the extrinsic's translation, which is metric, enters as `t_SC / s`.
//! With `t' = t_SC / s` and `M = R_scᵀ R_tᵀ`: `∂r/∂s = −s·g (M R_h − R_scᵀ) t'/s`,
//! and the extrinsic's `ρ` columns are divided by `s`.
//!
//! # Inactive cases
//!
//! A point at non-positive range or behind the target camera, or a pattern
//! pixel that leaves the target image (or its last row/column, where the
//! gradient is undefined), contributes nothing: zero residual and zero
//! Jacobian, as the line and patch factors do.
//!
//! # Parameter layout (7 blocks, 30 minimal DOF)
//!
//! ```text
//! params[0]: SE23 host        — 10D, 9 DOF
//! params[1]: SE23 target      — 10D, 9 DOF
//! params[2]: SE3  T_SC        —  7D, 6 DOF   body-from-camera
//! params[3]: Rn(1) ρ          —  1D
//! params[4]: Rn(2) (a, b) host
//! params[5]: Rn(2) (a, b) target
//! params[6]: Rn(1) s         scale of the visual frame
//! ```
//!
//! The factor whitens internally; register it with `NoiseModel::null()`.

use std::sync::Arc;

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::SE23;

use crate::core::variable::ManifoldVariable;
use crate::core::{CoreError, CoreResult};
use crate::factors::Factor;
use crate::factors::common::validate::expect_block_sizes;

/// Residuals per point: the 8-pixel pattern.
pub const PATTERN_PIXELS: usize = 8;
/// Columns of the Jacobian: two states, the extrinsic, `ρ`, two affine pairs
/// and the scale.
pub const JACOBIAN_COLUMNS: usize = 9 + 9 + 6 + 1 + 2 + 2 + 1;

/// A target image with gradients, shared by every factor that looks at it.
///
/// `data` holds `(I, ∂ₓI, ∂ᵧI)` per pixel, row-major, which is how a
/// photometric pyramid level stores it; pixel centres are at integer
/// coordinates.
#[derive(Clone, Debug)]
pub struct GradientImage {
    width: usize,
    height: usize,
    data: Arc<Vec<[f32; 3]>>,
}

impl GradientImage {
    /// Wrap shared pixel data.
    ///
    /// # Errors
    ///
    /// A size smaller than `2 × 2` or a length other than `width · height`.
    pub fn new(width: usize, height: usize, data: Arc<Vec<[f32; 3]>>) -> CoreResult<Self> {
        if width < 2 || height < 2 || data.len() != width * height {
            return Err(CoreError::InvalidInput(format!(
                "GradientImage: {} pixels for {width}x{height}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Bilinear `(I, ∂ₓI, ∂ᵧI)`; `None` outside the image or on its last row
    /// or column.
    #[must_use]
    pub fn sample(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        if !(x >= 0.0 && y >= 0.0 && x.is_finite() && y.is_finite()) {
            return None;
        }
        let (ix, iy) = (x.floor(), y.floor());
        let (xi, yi) = (ix as usize, iy as usize);
        if xi + 1 >= self.width || yi + 1 >= self.height {
            return None;
        }
        let (fx, fy) = (x - ix, y - iy);
        let base = xi + yi * self.width;
        let corners = [
            self.data[base],
            self.data[base + 1],
            self.data[base + self.width],
            self.data[base + self.width + 1],
        ];
        let w = [
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ];
        let mut out = [0.0; 3];
        for (k, value) in out.iter_mut().enumerate() {
            *value = (0..4).map(|c| w[c] * f64::from(corners[c][k])).sum();
        }
        Some(out)
    }
}

/// What a point carries from its host frame.
#[derive(Clone, Debug)]
pub struct PhotometricPoint {
    /// Host pixel `(u, v)` of the pattern centre.
    pub pixel: [f64; 2],
    /// Pattern offsets `δₖ` in pixels.
    pub offsets: [[f64; 2]; PATTERN_PIXELS],
    /// Host intensities at the pattern pixels.
    pub colors: [f32; PATTERN_PIXELS],
    /// Gradient weights `wₖ`.
    pub weights: [f32; PATTERN_PIXELS],
}

/// Weighting of the residual.
#[derive(Clone, Copy, Debug)]
pub struct PhotometricWeights {
    /// Huber threshold `γ`, grey levels.
    pub huber: f64,
    /// Scale `σ` in grey levels; residuals are divided by it.
    pub sigma: f64,
    /// DM-VIO's dynamic weight `W` (1 for none); the residual gets `√W`.
    pub dynamic: f64,
    /// Ratio of exposure times `t_target / t_host`.
    pub exposure_ratio: f64,
}

impl Default for PhotometricWeights {
    fn default() -> Self {
        Self {
            huber: 9.0,
            sigma: 1.0,
            dynamic: 1.0,
            exposure_ratio: 1.0,
        }
    }
}

/// The photometric residual of one point in one target frame.
pub struct PhotometricInverseRangeSe23Factor {
    point: PhotometricPoint,
    /// Pinhole `[fx, fy, cx, cy]`.
    k: [f64; 4],
    /// `b₀.z`: the optical-axis component of the unit bearing of the centre.
    bearing_z: f64,
    target: Arc<GradientImage>,
    weights: PhotometricWeights,
}

/// The variables the factor is evaluated at.
#[derive(Clone, Copy)]
struct State<'a> {
    scale: f64,
    host: &'a SE23,
    target: &'a SE23,
    extrinsic: &'a SE3,
    rho: f64,
    aff_h: [f64; 2],
    aff_t: [f64; 2],
}

/// Result of evaluating the factor at a state.
struct Evaluation {
    residual: [f64; PATTERN_PIXELS],
    /// Raw (unscaled) robust energy `Σ wₖ² hw r² (2 − hw)`.
    energy: f64,
    jacobian: Option<SMatrix<f64, PATTERN_PIXELS, JACOBIAN_COLUMNS>>,
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

impl PhotometricInverseRangeSe23Factor {
    /// The factor for `point` seen in `target`.
    ///
    /// # Errors
    ///
    /// Non-finite or non-positive intrinsics or weights.
    pub fn new(
        point: PhotometricPoint,
        k: [f64; 4],
        target: Arc<GradientImage>,
        weights: PhotometricWeights,
    ) -> CoreResult<Self> {
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(k[0]) || !positive(k[1]) || !k.iter().all(|v| v.is_finite()) {
            return Err(CoreError::InvalidInput(format!(
                "PhotometricInverseRangeSe23Factor: invalid intrinsics {k:?}"
            )));
        }
        if !positive(weights.huber)
            || !positive(weights.sigma)
            || !positive(weights.dynamic)
            || !positive(weights.exposure_ratio)
        {
            return Err(CoreError::InvalidInput(format!(
                "PhotometricInverseRangeSe23Factor: invalid weights {weights:?}"
            )));
        }
        let ray = Vector3::new(
            (point.pixel[0] - k[2]) / k[0],
            (point.pixel[1] - k[3]) / k[1],
            1.0,
        );
        Ok(Self {
            bearing_z: 1.0 / ray.norm(),
            point,
            k,
            target,
            weights,
        })
    }

    /// The raw robust energy of the point at a state: DSO's per-residual
    /// energy, in grey levels squared, for the outlier test. `None` when the
    /// factor is inactive there.
    #[must_use]
    pub fn energy(&self, params: &[&[f64]]) -> Option<f64> {
        let [host, target, extrinsic, rho, aff_h, aff_t, scale] = params else {
            return None;
        };
        let (host, target, extrinsic) = (
            SE23::from_param_slice(host),
            SE23::from_param_slice(target),
            SE3::from_param_slice(extrinsic),
        );
        self.evaluate(
            &State {
                scale: scale[0],
                host: &host,
                target: &target,
                extrinsic: &extrinsic,
                rho: rho[0],
                aff_h: [aff_h[0], aff_h[1]],
                aff_t: [aff_t[0], aff_t[1]],
            },
            false,
        )
        .map(|e| e.energy)
    }

    fn evaluate(&self, state: &State<'_>, with_jacobian: bool) -> Option<Evaluation> {
        let State {
            scale,
            host,
            target,
            extrinsic,
            rho,
            aff_h,
            aff_t,
        } = *state;
        if !rho.is_finite() || rho <= 0.0 || !(scale.is_finite() && scale > 0.0) {
            return None;
        }
        let [fx, fy, cx, cy] = self.k;
        let r_e = extrinsic.rotation_so3().rotation_matrix();
        let inv_scale = 1.0 / scale;
        let (t_e, r_h, p_h) = (
            extrinsic.translation() * inv_scale,
            host.rotation_matrix(),
            host.translation(),
        );
        let (r_t, p_t) = (target.rotation_matrix(), target.translation());
        let m = r_e.transpose() * r_t.transpose();
        let r_rel = m * r_h * r_e;
        let z0 = self.bearing_z / rho;
        let alpha = self.weights.exposure_ratio * (aff_t[0] - aff_h[0]).exp();
        let scale_base = self.weights.dynamic.sqrt() / self.weights.sigma;

        let mut residual = [0.0; PATTERN_PIXELS];
        let mut energy = 0.0;
        let mut jacobian =
            with_jacobian.then(SMatrix::<f64, PATTERN_PIXELS, JACOBIAN_COLUMNS>::zeros);

        for k in 0..PATTERN_PIXELS {
            let q = [
                self.point.pixel[0] + self.point.offsets[k][0],
                self.point.pixel[1] + self.point.offsets[k][1],
            ];
            let ray = Vector3::new((q[0] - cx) / fx, (q[1] - cy) / fy, 1.0);
            let p_host = z0 * ray;
            let u = r_e * p_host + t_e;
            let w = r_h * u + p_h;
            let v = r_t.transpose() * (w - p_t);
            let p_target = r_e.transpose() * (v - t_e);
            if p_target.z.is_nan() || p_target.z <= 1e-6 {
                return None;
            }
            let (x, y, z) = (p_target.x, p_target.y, p_target.z);
            let hit = self.target.sample(fx * x / z + cx, fy * y / z + cy)?;
            let host_term = f64::from(self.point.colors[k]) - aff_h[1];
            let raw = hit[0] - aff_t[1] - alpha * host_term;
            let hw = if raw.abs() < self.weights.huber {
                1.0
            } else {
                self.weights.huber / raw.abs()
            };
            let wk = f64::from(self.point.weights[k]);
            energy += wk * wk * hw * raw * raw * (2.0 - hw);
            let s = scale_base * hw.sqrt() * wk;
            residual[k] = s * raw;

            if let Some(jac) = jacobian.as_mut() {
                let g = Vector3::new(
                    hit[1] * fx / z,
                    hit[2] * fy / z,
                    -(hit[1] * fx * x + hit[2] * fy * y) / (z * z),
                );
                let g_m = (g.transpose() * m).transpose();
                let g_re = (g.transpose() * r_e.transpose()).transpose();
                let mut row = SVector::<f64, JACOBIAN_COLUMNS>::zeros();
                // host: ρ, θ (ν stays zero)
                let host_rho = (g_m.transpose() * r_h).transpose();
                let host_theta = -(g_m.transpose() * r_h * skew(&u)).transpose();
                row.fixed_rows_mut::<3>(0).copy_from(&host_rho);
                row.fixed_rows_mut::<3>(3).copy_from(&host_theta);
                // target
                row.fixed_rows_mut::<3>(9).copy_from(&(-g_re));
                row.fixed_rows_mut::<3>(12)
                    .copy_from(&(g_re.transpose() * skew(&v)).transpose());
                // extrinsic
                let gt_rel = g.transpose() * r_rel;
                row.fixed_rows_mut::<3>(18)
                    .copy_from(&((gt_rel - g.transpose()) * inv_scale).transpose());
                row.fixed_rows_mut::<3>(21).copy_from(
                    &(g.transpose() * skew(&p_target) - gt_rel * skew(&p_host)).transpose(),
                );
                // inverse range
                row[24] = -(gt_rel * p_host)[0] / rho;
                // scale: the extrinsic translation is t_SC / s.
                let d_t = (g.transpose() * (m * r_h - r_e.transpose())) * (-t_e * inv_scale);
                row[29] = d_t[0];
                for c in 0..25 {
                    row[c] *= s;
                }
                row[29] *= s;
                // affine: a_h, b_h, a_t, b_t
                let fixed = alpha * host_term;
                row[25] = s * fixed;
                row[26] = s * alpha;
                row[27] = -s * fixed;
                row[28] = -s;
                for c in 0..JACOBIAN_COLUMNS {
                    jac[(k, c)] = row[c];
                }
            }
        }
        Some(Evaluation {
            residual,
            energy,
            jacobian,
        })
    }
}

impl Factor for PhotometricInverseRangeSe23Factor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let [host, target, extrinsic, rho, aff_h, aff_t, scale] = params else {
            return;
        };
        let (host, target, extrinsic) = (
            SE23::from_param_slice(host),
            SE23::from_param_slice(target),
            SE3::from_param_slice(extrinsic),
        );
        let evaluation = self.evaluate(
            &State {
                scale: scale[0],
                host: &host,
                target: &target,
                extrinsic: &extrinsic,
                rho: rho[0],
                aff_h: [aff_h[0], aff_h[1]],
                aff_t: [aff_t[0], aff_t[1]],
            },
            jacobian.is_some(),
        );
        match evaluation {
            Some(e) => {
                residual[..PATTERN_PIXELS].copy_from_slice(&e.residual);
                if let (Some(mut jac), Some(values)) = (jacobian, e.jacobian) {
                    for row in 0..PATTERN_PIXELS {
                        for col in 0..JACOBIAN_COLUMNS {
                            *jac.rb_mut().get_mut(row, col) = values[(row, col)];
                        }
                    }
                }
            }
            None => {
                residual[..PATTERN_PIXELS].fill(0.0);
                if let Some(mut jac) = jacobian {
                    jac.fill(0.0);
                }
            }
        }
    }

    fn residual_dim(&self) -> usize {
        PATTERN_PIXELS
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (PATTERN_PIXELS, JACOBIAN_COLUMNS)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE, SE23::REP_SIZE, SE3::REP_SIZE, 1, 2, 2, 1],
            "PhotometricInverseRangeSe23Factor expects [SE23 host, SE23 target, SE3 T_SC, Rn(1) rho, Rn(2) aff_host, Rn(2) aff_target, Rn(1) scale]",
        )
    }
}

/// A prior on a frame's affine brightness parameters: `λ_a a² + λ_b b²`.
///
/// DSO adds it when exposure times are known, so the parameters stay near zero
/// (`a` is a log gain, `b` an offset in grey levels); without exposure times
/// the parameters absorb the exposure change and the prior is left out.
pub struct AffinePriorFactor {
    sqrt_lambda: [f64; 2],
}

impl AffinePriorFactor {
    /// A prior with weights `λ_a`, `λ_b`.
    ///
    /// # Errors
    ///
    /// A negative or non-finite weight.
    pub fn new(lambda_a: f64, lambda_b: f64) -> CoreResult<Self> {
        let ok = |x: f64| x.is_finite() && x >= 0.0;
        if !ok(lambda_a) || !ok(lambda_b) {
            return Err(CoreError::InvalidInput(format!(
                "AffinePriorFactor: invalid weights ({lambda_a}, {lambda_b})"
            )));
        }
        Ok(Self {
            sqrt_lambda: [lambda_a.sqrt(), lambda_b.sqrt()],
        })
    }
}

impl Factor for AffinePriorFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let [affine] = params else {
            return;
        };
        for i in 0..2 {
            residual[i] = self.sqrt_lambda[i] * affine[i];
        }
        if let Some(mut jac) = jacobian {
            jac.fill(0.0);
            for i in 0..2 {
                *jac.rb_mut().get_mut(i, i) = self.sqrt_lambda[i];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, 2)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(variables, &[2], "AffinePriorFactor expects [Rn(2) affine]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;
    use apex_manifolds::se23::SE23Tangent;
    use nalgebra::{DMatrix, UnitQuaternion};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    const K: [f64; 4] = [300.0, 300.0, 255.5, 255.5];
    const SIZE: usize = 512;
    const PLANE_Z: f64 = 4.0;
    const PATTERN: [[f64; 2]; 8] = [
        [0.0, -2.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [-2.0, 0.0],
        [0.0, 0.0],
        [2.0, 0.0],
        [-1.0, 1.0],
        [0.0, 2.0],
    ];

    fn texture(x: f64, y: f64, freq: f64) -> f64 {
        let (x, y) = (x * freq, y * freq);
        // Smooth on the scale of a pixel (a pixel is about 1 cm of the plane),
        // so the interpolated gradient the factor uses agrees with the
        // derivative of the interpolated intensity the finite differences see.
        128.0 + 120.0 * (x * 0.45).sin() * (y * 0.4).cos() + 70.0 * (x * 0.25 + y * 0.3).sin()
    }

    fn se23(t: Vector3<f64>, q: UnitQuaternion<f64>) -> SE23 {
        SE23::new(t, Vector3::new(0.1, -0.2, 0.3), q)
    }

    struct Scene {
        host: SE23,
        target: SE23,
        extrinsic: SE3,
        /// The first point and its true inverse range.
        rho: f64,
        point: PhotometricPoint,
        /// Every point with its true inverse range.
        points: Vec<(PhotometricPoint, f64)>,
        image: Arc<GradientImage>,
    }

    /// World ray through a camera pixel, for the body pose `x` and extrinsic `e`.
    fn ray(x: &SE23, e: &SE3, pixel: [f64; 2]) -> (Vector3<f64>, Vector3<f64>) {
        let r_wc = x.rotation_matrix() * e.rotation_so3().rotation_matrix();
        let origin = x.rotation_matrix() * e.translation() + x.translation();
        let dir = r_wc * Vector3::new((pixel[0] - K[2]) / K[0], (pixel[1] - K[3]) / K[1], 1.0);
        (origin, dir)
    }

    /// A gently curved surface `z = 4 + bump(x, y)`; a flat plane would leave
    /// the relative pose ambiguous.
    fn surface_height(x: f64, y: f64) -> f64 {
        PLANE_Z + 0.35 * (x * 1.7).sin() + 0.3 * (y * 1.3 + 0.5).cos()
    }

    /// Where the ray meets the surface, by fixed-point iteration on its length.
    fn surface_hit(origin: &Vector3<f64>, dir: &Vector3<f64>) -> Vector3<f64> {
        let mut s = (PLANE_Z - origin.z) / dir.z;
        for _ in 0..40 {
            let p = origin + s * dir;
            s = (surface_height(p.x, p.y) - origin.z) / dir.z;
        }
        origin + s * dir
    }

    fn plane_texture(origin: &Vector3<f64>, dir: &Vector3<f64>, freq: f64) -> f64 {
        let p = surface_hit(origin, dir);
        texture(p.x, p.y, freq)
    }

    fn scene() -> Result<Scene, Box<dyn std::error::Error>> {
        scene_with(&[[230.0, 270.0]], 1.0)
    }

    fn scene_with(pixels: &[[f64; 2]], freq: f64) -> Result<Scene, Box<dyn std::error::Error>> {
        let host = se23(
            Vector3::new(0.1, -0.2, 0.0),
            UnitQuaternion::from_euler_angles(0.03, -0.05, 0.1),
        );
        let target = se23(
            Vector3::new(0.45, -0.05, 0.12),
            UnitQuaternion::from_euler_angles(-0.02, 0.04, 0.13),
        );
        let extrinsic = SE3::new(
            Vector3::new(0.03, -0.05, 0.02),
            UnitQuaternion::from_euler_angles(0.01, -0.02, 0.015),
        );
        let r_wc = host.rotation_matrix() * extrinsic.rotation_so3().rotation_matrix();
        let points: Vec<(PhotometricPoint, f64)> = pixels
            .iter()
            .map(|&pixel| {
                let (origin, dir) = ray(&host, &extrinsic, pixel);
                let hit = surface_hit(&origin, &dir);
                let z_cam = (r_wc.transpose() * (hit - origin)).z;
                let bearing_z = 1.0
                    / Vector3::new((pixel[0] - K[2]) / K[0], (pixel[1] - K[3]) / K[1], 1.0).norm();
                let mut colors = [0.0f32; 8];
                for (c, o) in colors.iter_mut().zip(PATTERN) {
                    let (origin, dir) = ray(&host, &extrinsic, [pixel[0] + o[0], pixel[1] + o[1]]);
                    *c = plane_texture(&origin, &dir, freq) as f32;
                }
                (
                    PhotometricPoint {
                        pixel,
                        offsets: PATTERN,
                        colors,
                        weights: [1.0; 8],
                    },
                    bearing_z / z_cam,
                )
            })
            .collect();
        let intensity: Vec<f64> = (0..SIZE * SIZE)
            .map(|i| {
                let (origin, dir) =
                    ray(&target, &extrinsic, [(i % SIZE) as f64, (i / SIZE) as f64]);
                plane_texture(&origin, &dir, freq)
            })
            .collect();
        let mut data = vec![[0.0f32; 3]; SIZE * SIZE];
        for y in 0..SIZE {
            for x in 0..SIZE {
                let i = x + y * SIZE;
                data[i][0] = intensity[i] as f32;
                if x > 0 && y > 0 && x + 1 < SIZE && y + 1 < SIZE {
                    data[i][1] = (0.5 * (intensity[i + 1] - intensity[i - 1])) as f32;
                    data[i][2] = (0.5 * (intensity[i + SIZE] - intensity[i - SIZE])) as f32;
                }
            }
        }
        Ok(Scene {
            host,
            target,
            extrinsic,
            rho: points[0].1,
            point: points[0].0.clone(),
            points,
            image: Arc::new(GradientImage::new(SIZE, SIZE, Arc::new(data))?),
        })
    }

    fn factor_for(
        s: &Scene,
        point: &PhotometricPoint,
    ) -> Result<PhotometricInverseRangeSe23Factor, Box<dyn std::error::Error>> {
        Ok(PhotometricInverseRangeSe23Factor::new(
            point.clone(),
            K,
            Arc::clone(&s.image),
            PhotometricWeights::default(),
        )?)
    }

    fn factor(
        s: &Scene,
        weights: PhotometricWeights,
    ) -> Result<PhotometricInverseRangeSe23Factor, Box<dyn std::error::Error>> {
        Ok(PhotometricInverseRangeSe23Factor::new(
            s.point.clone(),
            K,
            Arc::clone(&s.image),
            weights,
        )?)
    }

    #[derive(Clone)]
    struct Vars {
        host: SE23,
        target: SE23,
        extrinsic: SE3,
        rho: f64,
        aff_h: [f64; 2],
        aff_t: [f64; 2],
        scale: f64,
    }

    fn vars(s: &Scene) -> Vars {
        Vars {
            host: s.host.clone(),
            target: s.target.clone(),
            extrinsic: s.extrinsic.clone(),
            rho: s.rho,
            aff_h: [0.0, 0.0],
            aff_t: [0.0, 0.0],
            scale: 1.0,
        }
    }

    fn params(v: &Vars) -> [Vec<f64>; 7] {
        [
            v.host.as_param_slice().to_vec(),
            v.target.as_param_slice().to_vec(),
            v.extrinsic.as_param_slice().to_vec(),
            vec![v.rho],
            v.aff_h.to_vec(),
            v.aff_t.to_vec(),
            vec![v.scale],
        ]
    }

    fn run(f: &PhotometricInverseRangeSe23Factor, v: &Vars) -> (Vec<f64>, DMatrix<f64>) {
        let p = params(v);
        let refs: Vec<&[f64]> = p.iter().map(Vec::as_slice).collect();
        let mut residual = vec![0.0; 8];
        let mut buf = vec![0.0; 8 * JACOBIAN_COLUMNS];
        let jac = faer::mat::MatMut::from_column_major_slice_mut(&mut buf, 8, JACOBIAN_COLUMNS);
        f.linearize(&refs, &mut residual, Some(jac));
        (
            residual,
            DMatrix::from_column_slice(8, JACOBIAN_COLUMNS, &buf),
        )
    }

    fn residual_only(f: &PhotometricInverseRangeSe23Factor, v: &Vars) -> Vec<f64> {
        let p = params(v);
        let refs: Vec<&[f64]> = p.iter().map(Vec::as_slice).collect();
        let mut residual = vec![0.0; 8];
        f.linearize(&refs, &mut residual, None);
        residual
    }

    /// Which block of the factor a Jacobian column belongs to.
    #[derive(Clone, Copy)]
    enum Column {
        Host(usize),
        Target(usize),
        Extrinsic(usize),
        Rho,
        AffineHost(usize),
        AffineTarget(usize),
        Scale,
    }

    fn perturbed(v: &Vars, column: Column, d: f64) -> Vars {
        let mut out = v.clone();
        let se23_step = |k: usize| {
            let mut t = [0.0; 9];
            t[k] = d;
            SE23Tangent::from_slice(&t)
        };
        match column {
            Column::Host(k) => out.host = v.host.right_plus(&se23_step(k), None, None),
            Column::Target(k) => out.target = v.target.right_plus(&se23_step(k), None, None),
            Column::Extrinsic(k) => {
                let mut t = [0.0; 6];
                t[k] = d;
                out.extrinsic = v
                    .extrinsic
                    .right_plus(&SE3Tangent::from_slice(&t), None, None);
            }
            Column::Rho => out.rho += d,
            Column::AffineHost(k) => out.aff_h[k] += d,
            Column::AffineTarget(k) => out.aff_t[k] += d,
            Column::Scale => out.scale += d,
        }
        out
    }

    #[test]
    fn residual_is_small_at_the_true_state_and_grows_with_error() -> TestResult {
        let s = scene()?;
        assert!(s.rho > 0.0);
        let f = factor(&s, PhotometricWeights::default())?;
        let (r, _) = run(&f, &vars(&s));
        assert!(r.iter().any(|x| *x != 0.0), "the factor must be active");
        assert!(r.iter().all(|x| x.abs() < 1.5), "{r:?}");
        let mut off = vars(&s);
        off.rho *= 1.3;
        let (r_off, _) = run(&f, &off);
        let size = |r: &[f64]| r.iter().map(|x| x * x).sum::<f64>();
        assert!(size(&r_off) > 20.0 * size(&r) + 100.0);
        Ok(())
    }

    #[test]
    fn jacobian_matches_central_differences_on_every_block() -> TestResult {
        let s = scene()?;
        let f = factor(&s, PhotometricWeights::default())?;
        let mut base = vars(&s);
        // Away from the truth so the residuals are far from zero.
        base.host = base.host.right_plus(
            &SE23Tangent::from_slice(&[0.01, -0.008, 0.012, 0.004, -0.003, 0.002, 0.0, 0.0, 0.0]),
            None,
            None,
        );
        base.rho *= 1.05;
        base.aff_h = [0.02, 1.5];
        base.aff_t = [-0.03, -2.0];
        base.scale = 1.07;
        let (base_r, jac) = run(&f, &base);
        assert!(
            base_r.iter().all(|x| x.abs() > 1e-3),
            "active away from the truth"
        );
        assert!(jac.norm() > 1.0);

        let mut columns: Vec<(usize, Column)> = Vec::new();
        for k in 0..9 {
            columns.push((k, Column::Host(k)));
            columns.push((9 + k, Column::Target(k)));
        }
        for k in 0..6 {
            columns.push((18 + k, Column::Extrinsic(k)));
        }
        columns.push((24, Column::Rho));
        columns.push((25, Column::AffineHost(0)));
        columns.push((26, Column::AffineHost(1)));
        columns.push((27, Column::AffineTarget(0)));
        columns.push((28, Column::AffineTarget(1)));
        columns.push((29, Column::Scale));

        let mut checked = 0;
        for (col, which) in columns {
            // The image is stored as f32, so a very small step would measure
            // its rounding; the interpolant is piecewise linear, so a step of a
            // hundredth of a pixel is still exact.
            let step = 1e-4;
            let plus = residual_only(&f, &perturbed(&base, which, step));
            let minus = residual_only(&f, &perturbed(&base, which, -step));
            let analytic = jac.column(col);
            let numeric: Vec<f64> = (0..8)
                .map(|i| (plus[i] - minus[i]) / (2.0 * step))
                .collect();
            let n = numeric.iter().map(|x| x * x).sum::<f64>().sqrt();
            let diff = (0..8)
                .map(|i| (analytic[i] - numeric[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            if matches!(col, 6..=8 | 15..=17) {
                assert!(analytic.norm() < 1e-12 && n < 1e-6, "velocity column {col}");
                continue;
            }
            assert!(
                diff <= 0.05 * n.max(1.0),
                "column {col}: analytic {:.4} numeric {n:.4} difference {diff:.4}",
                analytic.norm()
            );
            checked += 1;
        }
        assert_eq!(checked, 24);
        Ok(())
    }

    #[test]
    fn the_residual_does_not_change_under_a_rigid_motion_of_the_world() -> TestResult {
        let s = scene()?;
        let f = factor(&s, PhotometricWeights::default())?;
        let base = vars(&s);
        let q = UnitQuaternion::from_euler_angles(0.2, -0.3, 0.9);
        let shift = Vector3::new(3.0, -2.0, 1.0);
        let moved = |x: &SE23| {
            SE23::new(
                q * x.translation() + shift,
                q * x.velocity(),
                q * x.rotation_quaternion(),
            )
        };
        let mut other = base.clone();
        other.host = moved(&base.host);
        other.target = moved(&base.target);
        let (a, b) = (residual_only(&f, &base), residual_only(&f, &other));
        for i in 0..8 {
            assert!((a[i] - b[i]).abs() < 1e-9);
        }
        Ok(())
    }

    #[test]
    fn inactive_cases_give_zero_residual_and_zero_jacobian() -> TestResult {
        let s = scene()?;
        let f = factor(&s, PhotometricWeights::default())?;
        let mut behind = vars(&s);
        behind.target = se23(Vector3::new(0.0, 0.0, 30.0), UnitQuaternion::identity());
        let mut negative = vars(&s);
        negative.rho = -1.0;
        let mut outside = vars(&s);
        outside.target = se23(Vector3::new(40.0, 0.0, 0.0), UnitQuaternion::identity());
        for v in [behind, negative, outside] {
            let (r, j) = run(&f, &v);
            assert!(r.iter().all(|x| *x == 0.0));
            assert!(j.iter().all(|x| *x == 0.0));
        }
        Ok(())
    }

    #[test]
    fn weights_scale_the_residual() -> TestResult {
        let s = scene()?;
        let plain = factor(&s, PhotometricWeights::default())?;
        let mut off = vars(&s);
        off.aff_t = [0.0, 3.0];
        let (r1, _) = run(&plain, &off);
        let dynamic = factor(
            &s,
            PhotometricWeights {
                dynamic: 0.25,
                ..PhotometricWeights::default()
            },
        )?;
        let (r2, _) = run(&dynamic, &off);
        let sigma = factor(
            &s,
            PhotometricWeights {
                sigma: 2.0,
                ..PhotometricWeights::default()
            },
        )?;
        let (r3, _) = run(&sigma, &off);
        for i in 0..8 {
            assert!((r2[i] - 0.5 * r1[i]).abs() < 1e-9);
            assert!((r3[i] - 0.5 * r1[i]).abs() < 1e-9);
        }
        // Huber: a raw residual far above the threshold is scaled by sqrt(hw).
        let mut far = vars(&s);
        far.aff_t = [0.0, 60.0];
        let (r4, _) = run(&plain, &far);
        assert!(r4.iter().all(|x| x.abs() < 60.0 * 0.5), "{r4:?}");
        let p = params(&far);
        let refs: Vec<&[f64]> = p.iter().map(Vec::as_slice).collect();
        let energy = plain.energy(&refs).ok_or("active")?;
        assert!(energy > 8.0 * 9.0 * 9.0 * 0.5);
        Ok(())
    }

    #[test]
    fn bad_inputs_are_refused() -> TestResult {
        let s = scene()?;
        let bad_k = PhotometricInverseRangeSe23Factor::new(
            s.point.clone(),
            [0.0, 300.0, 255.0, 255.0],
            Arc::clone(&s.image),
            PhotometricWeights::default(),
        );
        assert!(bad_k.is_err());
        let bad_weight = PhotometricInverseRangeSe23Factor::new(
            s.point.clone(),
            K,
            Arc::clone(&s.image),
            PhotometricWeights {
                sigma: 0.0,
                ..PhotometricWeights::default()
            },
        );
        assert!(bad_weight.is_err());
        assert!(GradientImage::new(1, 1, Arc::new(vec![[0.0; 3]])).is_err());
        assert!(GradientImage::new(4, 4, Arc::new(vec![[0.0; 3]; 3])).is_err());
        Ok(())
    }

    #[test]
    fn the_affine_prior_pulls_toward_zero() -> TestResult {
        let prior = AffinePriorFactor::new(4.0, 0.25)?;
        let mut r = [0.0; 2];
        let mut buf = [0.0; 4];
        let jac = faer::mat::MatMut::from_column_major_slice_mut(&mut buf, 2, 2);
        prior.linearize(&[&[0.5, 2.0]], &mut r, Some(jac));
        assert!((r[0] - 1.0).abs() < 1e-12 && (r[1] - 1.0).abs() < 1e-12);
        assert_eq!(buf, [2.0, 0.0, 0.0, 0.5]);
        assert!(AffinePriorFactor::new(-1.0, 0.0).is_err());
        Ok(())
    }

    /// Many points, a free target pose and free inverse ranges: LM finds the
    /// true pose and depths from a start a couple of pixels off.
    #[test]
    fn lm_recovers_the_pose_and_the_depths() -> TestResult {
        use crate::JacobianMode;
        use crate::core::noise::NoiseModel;
        use crate::core::problem::Problem;
        use crate::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
        use apex_manifolds::ManifoldType;
        use nalgebra::DVector;

        let pixels: Vec<[f64; 2]> = (0..6)
            .flat_map(|j| {
                (0..6).map(move |i| [90.0 + 66.0 * f64::from(i), 100.0 + 60.0 * f64::from(j)])
            })
            .collect();
        let s = scene_with(&pixels, 3.0)?;
        let mut problem = Problem::new(JacobianMode::Sparse);
        let fixed = |problem: &mut Problem, ty: ManifoldType, values: &[f64]| {
            let key = problem.add_variable(ty, DVector::from_column_slice(values));
            let dof = match ty {
                ManifoldType::SE23 => 9,
                ManifoldType::SE3 => 6,
                _ => values.len(),
            };
            for i in 0..dof {
                problem.fix_variable(key, i);
            }
            key
        };
        let host = fixed(&mut problem, ManifoldType::SE23, s.host.as_param_slice());
        let extrinsic = fixed(
            &mut problem,
            ManifoldType::SE3,
            s.extrinsic.as_param_slice(),
        );
        let aff_h = fixed(&mut problem, ManifoldType::RN, &[0.0, 0.0]);
        let aff_t = fixed(&mut problem, ManifoldType::RN, &[0.0, 0.0]);
        let scale = fixed(&mut problem, ManifoldType::RN, &[1.0]);

        let start = s.target.right_plus(
            &SE23Tangent::from_slice(&[0.015, -0.01, 0.012, 0.004, -0.003, 0.005, 0.0, 0.0, 0.0]),
            None,
            None,
        );
        let target = problem.add_variable(
            ManifoldType::SE23,
            DVector::from_column_slice(start.as_param_slice()),
        );
        for i in 6..9 {
            problem.fix_variable(target, i);
        }
        let mut rho_keys = Vec::new();
        for (n, (point, rho)) in s.points.iter().enumerate() {
            // One depth is held at its true value: with a fixed host and free
            // depths a monocular pair fixes the translation only up to scale.
            let key = if n == 0 {
                fixed(&mut problem, ManifoldType::RN, &[*rho])
            } else {
                problem.add_variable(ManifoldType::RN, DVector::from_element(1, rho * 1.08))
            };
            if n > 0 {
                problem.mark_for_elimination(key);
            }
            rho_keys.push(key);
            let f = PhotometricInverseRangeSe23Factor::new(
                point.clone(),
                K,
                Arc::clone(&s.image),
                PhotometricWeights::default(),
            )?;
            problem.add_residual_block_with_noise(
                &[host, target, extrinsic, key, aff_h, aff_t, scale],
                Box::new(f),
                None,
                NoiseModel::null(),
            );
        }
        let config = LevenbergMarquardtConfig::for_bundle_adjustment().with_max_iterations(60);
        let result = LevenbergMarquardt::with_config(config).optimize(&mut problem)?;
        // The cost at the true state: the floor set by interpolation error.
        let truth_cost: f64 = s
            .points
            .iter()
            .map(|(point, rho)| {
                let f = factor_for(&s, point)?;
                let v = Vars {
                    rho: *rho,
                    ..vars(&s)
                };
                Ok(0.5 * residual_only(&f, &v).iter().map(|r| r * r).sum::<f64>())
            })
            .sum::<Result<f64, Box<dyn std::error::Error>>>()?;
        assert!(
            result.final_cost < 0.01 * result.initial_cost,
            "{} -> {}",
            result.initial_cost,
            result.final_cost
        );
        assert!(
            result.final_cost < 2.0 * truth_cost + 1.0,
            "converged to {} against {truth_cost} at the truth",
            result.final_cost
        );

        // The pose is only weakly observable here (a few centimetres of
        // translation trade against a fraction of a degree of rotation at
        // essentially equal cost), so the check is that it has not run away.
        let solved = SE23::from_param_slice(
            result
                .parameters
                .get(target)
                .ok_or("target")?
                .as_param_slice(),
        );
        let dt = (solved.translation() - s.target.translation()).norm();
        let dr = (solved.rotation_quaternion().inverse() * s.target.rotation_quaternion()).angle();
        assert!(dt < 0.05, "position error {dt}");
        assert!(dr < 0.01, "rotation error {dr}");
        Ok(())
    }
}
