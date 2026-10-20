//! Relative pose between two keyframes in four degrees of freedom: position and
//! yaw. Roll and pitch are not variables.
//!
//! A visual-inertial odometry observes gravity, so its roll and pitch do not
//! drift; only position and the heading about gravity do. A pose graph that
//! lets a loop closure bend roll and pitch too would trade away what the
//! odometry knows for noise in the loop (VINS-Mono's 4-DoF graph).
//!
//! Each node is an `Rn(4)` variable `[x, y, z, ψ]`. Its tilt (roll and pitch) is
//! a constant of the factor, so the node's rotation is `R_i = Rz(ψ_i)·Tilt_i`:
//!
//! ```text
//! r_t = R_iᵀ (p_j − p_i) − Δp            3 rows, in node i's frame
//! r_ψ = wrap(ψ_j − ψ_i − Δψ)             1 row
//! ```
//!
//! The residual is raw: register the factor with the noise model that whitens
//! `[Δp (m), Δψ (rad)]`.

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, Vector3};

/// Relative position and yaw between two `Rn(4)` pose nodes.
#[derive(Debug, Clone)]
pub struct Relative4DofFactor {
    /// Node i's roll and pitch: `R_i = Rz(ψ_i)·tilt_i`.
    tilt_i: Matrix3<f64>,
    /// Measured position of j in node i's frame.
    delta_position: Vector3<f64>,
    /// Measured yaw of j relative to i \[rad\].
    delta_yaw: f64,
}

impl Relative4DofFactor {
    /// A factor with the measurement `delta_position` (j in i's frame) and
    /// `delta_yaw`, for a node i whose roll and pitch are `tilt_i`.
    #[must_use]
    pub fn new(tilt_i: Matrix3<f64>, delta_position: Vector3<f64>, delta_yaw: f64) -> Self {
        Self {
            tilt_i,
            delta_position,
            delta_yaw,
        }
    }
}

/// `Rz(ψ)`.
fn rot_z(yaw: f64) -> Matrix3<f64> {
    let (s, c) = yaw.sin_cos();
    Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}

/// An angle wrapped to `(-π, π]`.
fn wrap(angle: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let wrapped = (angle + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI;
    if wrapped <= -std::f64::consts::PI {
        std::f64::consts::PI
    } else {
        wrapped
    }
}

impl Factor for Relative4DofFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let (pi, pj) = (&params[0], &params[1]);
        let (p_i, p_j) = (
            Vector3::new(pi[0], pi[1], pi[2]),
            Vector3::new(pj[0], pj[1], pj[2]),
        );
        let (yaw_i, yaw_j) = (pi[3], pj[3]);

        // R_iᵀ d = tiltᵀ · Rz(ψ_i)ᵀ · d
        let d = p_j - p_i;
        let u = rot_z(yaw_i).transpose() * d;
        let r_t = self.tilt_i.transpose() * u - self.delta_position;
        residual[0] = r_t.x;
        residual[1] = r_t.y;
        residual[2] = r_t.z;
        residual[3] = wrap(yaw_j - yaw_i - self.delta_yaw);

        let Some(mut jac) = jacobian else { return };
        let r_i_t = self.tilt_i.transpose() * rot_z(yaw_i).transpose();
        // d(Rz(ψ)ᵀ d)/dψ = -e_z × (Rz(ψ)ᵀ d)
        let dyaw = -(self.tilt_i.transpose() * Vector3::z().cross(&u));
        for row in 0..3 {
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, col) = -r_i_t[(row, col)];
                *jac.rb_mut().get_mut(row, 4 + col) = r_i_t[(row, col)];
            }
            *jac.rb_mut().get_mut(row, 3) = dyaw[row];
        }
        *jac.rb_mut().get_mut(3, 3) = -1.0;
        *jac.rb_mut().get_mut(3, 7) = 1.0;
    }

    fn residual_dim(&self) -> usize {
        4
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (4, 8)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 2
            || variables
                .iter()
                .any(|v| v.manifold_type_name() != "Rn" || v.as_param_slice().len() != 4)
        {
            return Err("Relative4DofFactor connects two Rn(4) variables".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn tilt(roll: f64, pitch: f64) -> Matrix3<f64> {
        let (sr, cr) = roll.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, cr, -sr, 0.0, sr, cr);
        let ry = Matrix3::new(cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp);
        ry * rx
    }

    fn evaluate(factor: &Relative4DofFactor, a: &[f64], b: &[f64]) -> [f64; 4] {
        let mut r = [0.0; 4];
        factor.linearize(&[a, b], &mut r, None);
        r
    }

    #[test]
    fn the_residual_vanishes_at_the_measurement() {
        let t = tilt(0.1, -0.2);
        let a = [1.0, 2.0, 3.0, 0.7];
        let r_i = rot_z(a[3]) * t;
        let delta = Vector3::new(0.4, -0.3, 0.2);
        let p_j = Vector3::new(a[0], a[1], a[2]) + r_i * delta;
        let b = [p_j.x, p_j.y, p_j.z, 0.7 + 0.25];
        let factor = Relative4DofFactor::new(t, delta, 0.25);
        let r = evaluate(&factor, &a, &b);
        assert!(r.iter().all(|v| v.abs() < 1e-12), "{r:?}");
    }

    #[test]
    fn the_yaw_residual_wraps() {
        let factor = Relative4DofFactor::new(Matrix3::identity(), Vector3::zeros(), 0.0);
        let r = evaluate(&factor, &[0.0, 0.0, 0.0, 3.0], &[0.0, 0.0, 0.0, -3.0]);
        assert!(
            (r[3] - (-6.0 + std::f64::consts::TAU)).abs() < 1e-12,
            "{}",
            r[3]
        );
    }

    #[test]
    fn the_jacobian_matches_finite_differences() {
        let factor = Relative4DofFactor::new(tilt(0.15, 0.1), Vector3::new(0.5, 0.2, -0.1), 0.3);
        let a = [0.3, -0.4, 0.8, 1.1];
        let b = [1.1, 0.5, 0.6, 1.5];
        let mut r = [0.0; 4];
        let mut jac = Mat::<f64>::zeros(4, 8);
        factor.linearize(&[&a, &b], &mut r, Some(jac.as_mut()));
        let step = 1e-6;
        for col in 0..8 {
            let (mut plus_a, mut plus_b) = (a, b);
            let (mut minus_a, mut minus_b) = (a, b);
            if col < 4 {
                plus_a[col] += step;
                minus_a[col] -= step;
            } else {
                plus_b[col - 4] += step;
                minus_b[col - 4] -= step;
            }
            let (rp, rm) = (
                evaluate(&factor, &plus_a, &plus_b),
                evaluate(&factor, &minus_a, &minus_b),
            );
            for row in 0..4 {
                let numeric = (rp[row] - rm[row]) / (2.0 * step);
                assert!(
                    (numeric - jac[(row, col)]).abs() < 1e-6,
                    "row {row} col {col}: {numeric} vs {}",
                    jac[(row, col)]
                );
            }
        }
    }

    #[test]
    fn it_wants_two_rn4_variables() {
        let factor = Relative4DofFactor::new(Matrix3::identity(), Vector3::zeros(), 0.0);
        assert!(factor.validate_variables(&[]).is_err());
    }
}
