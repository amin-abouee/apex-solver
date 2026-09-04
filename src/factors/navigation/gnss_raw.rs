//! Raw GNSS pseudorange and Doppler (range-rate) factors
//! (GTSAM `PseudorangeFactor` / `DopplerFactor` analogues).
//!
//! # Pseudorange
//!
//! ```text
//! r = ‖s − x‖ + b − ρ_measured        (1D, meters)
//! ```
//! with receiver position `x`, receiver clock bias `b` (in **meters**, i.e.
//! already range-equivalent), satellite position `s` and measured
//! pseudorange `ρ` fixed in the factor.
//!
//! # Doppler (range rate)
//!
//! ```text
//! r = f̂ᵀ·(v_s − v_r) − ρ̇_measured    (1D, m/s)
//! f̂ = (s − x)/‖s − x‖
//! ```
//! over receiver position and velocity; satellite position/velocity are
//! fixed ephemeris inputs.

use faer::prelude::ReborrowMut;
use nalgebra::Vector3;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// Raw pseudorange factor over `[receiver position (3), clock bias (1)]`.
#[derive(Clone)]
pub struct PseudorangeFactor {
    /// Satellite position (ECEF or the graph's global frame) [m].
    pub satellite_position: Vector3<f64>,
    /// Measured pseudorange [m] (includes the receiver clock bias).
    pub measured_range: f64,
}

impl PseudorangeFactor {
    /// Create the factor from satellite position and measured pseudorange.
    pub fn new(satellite_position: Vector3<f64>, measured_range: f64) -> Self {
        Self {
            satellite_position,
            measured_range,
        }
    }
}

impl Factor for PseudorangeFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            2,
            "PseudorangeFactor expects [position, clock bias]"
        );
        debug_assert_eq!(params[0].len(), 3, "params[0] must be a 3D position");
        debug_assert_eq!(params[1].len(), 1, "params[1] must be a scalar clock bias");

        let pos = Vector3::new(params[0][0], params[0][1], params[0][2]);
        let bias = params[1][0];
        let delta = pos - self.satellite_position;
        let dist = delta.norm();

        if dist < 1e-9 {
            // Receiver exactly at the satellite: geometry is undefined.
            // Bounded constant penalty keeps the factor from becoming free.
            residual[0] = 1.0e6;
            if let Some(mut jac) = jacobian {
                for c in 0..4 {
                    *jac.rb_mut().get_mut(0, c) = 0.0;
                }
            }
            return;
        }

        residual[0] = dist + bias - self.measured_range;

        let Some(mut jac) = jacobian else { return };
        let d_dist_d_pos = delta / dist; // 1×3
        for col in 0..3 {
            *jac.rb_mut().get_mut(0, col) = d_dist_d_pos[col];
        }
        *jac.rb_mut().get_mut(0, 3) = 1.0;
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 4)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 2
            || variables[0].as_param_slice().len() != 3
            || variables[1].as_param_slice().len() != 1
        {
            return Err("PseudorangeFactor expects [R³ position, R¹ clock bias]".into());
        }
        Ok(())
    }
}

/// Doppler range-rate factor over `[receiver position (3), receiver velocity (3)]`.
#[derive(Clone)]
pub struct DopplerFactor {
    /// Satellite position (fixed ephemeris) [m].
    pub satellite_position: Vector3<f64>,
    /// Satellite velocity (fixed ephemeris) [m/s].
    pub satellite_velocity: Vector3<f64>,
    /// Measured range rate [m/s] (positive: range increasing).
    pub measured_range_rate: f64,
}

impl DopplerFactor {
    /// Create the factor from fixed satellite ephemeris and the measured
    /// range rate.
    pub fn new(
        satellite_position: Vector3<f64>,
        satellite_velocity: Vector3<f64>,
        measured_range_rate: f64,
    ) -> Self {
        Self {
            satellite_position,
            satellite_velocity,
            measured_range_rate,
        }
    }
}

impl Factor for DopplerFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(
            params.len(),
            2,
            "DopplerFactor expects [position, velocity]"
        );
        debug_assert_eq!(params[0].len(), 3, "params[0] must be a 3D position");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be a 3D velocity");

        let pos = Vector3::new(params[0][0], params[0][1], params[0][2]);
        let vel = Vector3::new(params[1][0], params[1][1], params[1][2]);
        let delta = self.satellite_position - pos;
        let dist = delta.norm();

        if dist < 1e-9 {
            residual[0] = 1.0e6;
            if let Some(mut jac) = jacobian {
                for c in 0..6 {
                    *jac.rb_mut().get_mut(0, c) = 0.0;
                }
            }
            return;
        }

        let f_hat = delta / dist;
        let rel_vel = self.satellite_velocity - vel;
        let range_rate = f_hat.dot(&rel_vel);
        residual[0] = range_rate - self.measured_range_rate;

        let Some(mut jac) = jacobian else { return };

        // ∂(f̂ᵀΔv)/∂vel_r = −f̂ᵀ
        for col in 0..3 {
            *jac.rb_mut().get_mut(0, 3 + col) = -f_hat[col];
        }

        // ∂/∂pos: f̂ᵀΔv with f̂ = (s − x)/d:
        // df̂ᵀ/dx = −(I − f̂f̂ᵀ)/d  →  d(range_rate)/dx = −(Δv − (f̂ᵀΔv)f̂)ᵀ/d
        let proj = rel_vel - f_hat * range_rate;
        for col in 0..3 {
            *jac.rb_mut().get_mut(0, col) = -proj[col] / dist;
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 6)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 2
            || variables[0].as_param_slice().len() != 3
            || variables[1].as_param_slice().len() != 3
        {
            return Err("DopplerFactor expects [R³ position, R³ velocity]".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd_check<F>(
        f: &F,
        params: &[Vec<f64>],
        rows: usize,
        cols: usize,
        tol: f64,
    ) -> Result<(), String>
    where
        F: Fn(&[&[f64]], &mut [f64], Option<faer::mat::MatMut<'_, f64>>),
    {
        let slices: Vec<&[f64]> = params.iter().map(|p| p.as_slice()).collect();
        let mut r0 = vec![0.0; rows];
        let mut jac_buf = vec![0.0; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        f(&slices, &mut r0, Some(jac_mut));

        const EPS: f64 = 1.0;
        let mut col_offset = 0;
        for (block_idx, block_len) in [(0usize, 3usize), (1, params[1].len())] {
            for col in 0..block_len {
                let mut plus = params.to_vec();
                let mut minus = params.to_vec();
                plus[block_idx][col] += EPS;
                minus[block_idx][col] -= EPS;
                let plus_slices: Vec<&[f64]> = plus.iter().map(|p| p.as_slice()).collect();
                let minus_slices: Vec<&[f64]> = minus.iter().map(|p| p.as_slice()).collect();
                let mut r_plus = vec![0.0; rows];
                let mut r_minus = vec![0.0; rows];
                f(&plus_slices, &mut r_plus, None);
                f(&minus_slices, &mut r_minus, None);
                for row in 0..rows {
                    let fd = (r_plus[row] - r_minus[row]) / (2.0 * EPS);
                    let ana = jac_buf[(col_offset + col) * rows + row];
                    if (fd - ana).abs() > tol {
                        return Err(format!(
                            "J[{row},{}]: analytical={ana:.6} fd={fd:.6}",
                            col_offset + col
                        ));
                    }
                }
            }
            col_offset += block_len;
        }
        Ok(())
    }

    #[test]
    fn pseudorange_zero_residual_and_fd() -> Result<(), String> {
        let sat = Vector3::new(1.9e7, 0.0, 1.2e7);
        let pos = Vector3::new(0.0, 0.0, 0.0);
        let geo_range = (sat - pos).norm();
        let bias = 50.0;
        let factor = PseudorangeFactor::new(sat, geo_range + bias);

        let pos_v = vec![0.0, 0.0, 0.0];
        let bias_v = vec![bias];
        let mut residual = vec![0.0; 1];
        factor.linearize(&[&pos_v, &bias_v], &mut residual, None);
        assert!(residual[0].abs() < 1e-6, "residual = {}", residual[0]);

        // Position Jacobian w.r.t. large coordinates needs EPS ≫ 1 m — the
        // helper uses EPS = 1.
        fd_check(
            &|p, r, j| factor.linearize(p, r, j),
            &[pos_v.clone(), bias_v.clone()],
            1,
            4,
            1e-4,
        )?;
        Ok(())
    }

    #[test]
    fn pseudorange_degenerate_geometry_is_penalized() {
        let factor = PseudorangeFactor::new(Vector3::new(1.0, 2.0, 3.0), 10.0);
        let pos_v = vec![1.0, 2.0, 3.0]; // exactly at the satellite
        let bias_v = vec![0.0];
        let mut residual = vec![0.0; 1];
        factor.linearize(&[&pos_v, &bias_v], &mut residual, None);
        assert!(residual[0] > 1.0e5, "expected penalty, got {}", residual[0]);
    }

    #[test]
    fn doppler_zero_residual_and_fd() -> Result<(), String> {
        let sat_pos = Vector3::new(2.0e7, 1.0e6, 1.5e7);
        let sat_vel = Vector3::new(100.0, -500.0, 800.0);
        let pos = Vector3::new(0.0, 0.0, 0.0);
        let vel = Vector3::new(1.0, 2.0, -3.0);
        let f_hat = (sat_pos - pos).normalize();
        let measured = f_hat.dot(&(sat_vel - vel));
        let factor = DopplerFactor::new(sat_pos, sat_vel, measured);

        let pos_v = vec![0.0, 0.0, 0.0];
        let vel_v = vec![1.0, 2.0, -3.0];
        let mut residual = vec![0.0; 1];
        factor.linearize(&[&pos_v, &vel_v], &mut residual, None);
        assert!(residual[0].abs() < 1e-9, "residual = {}", residual[0]);

        fd_check(
            &|p, r, j| factor.linearize(p, r, j),
            &[pos_v, vel_v],
            1,
            6,
            1e-4,
        )?;
        Ok(())
    }
}
