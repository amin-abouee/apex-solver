//! Homogeneous point error factor for landmark constraints.
//!
//! Implements a unary constraint on a homogeneous 4D landmark point,
//! penalizing deviation from a measured 3D Euclidean position.
//!
//! # Mathematical Formulation
//!
//! Given a homogeneous point `hp = [x, y, z, w]ᵀ ∈ R⁴`, the dehomogenized
//! 3D position is `p_est = hp[0:3] / hp[3]`.
//!
//! ```text
//! e = sqrt_info · (p_meas − p_est)   ∈ R³
//! ```
//!
//! # Parameter Layout (1 block, 4D)
//!
//! ```text
//! params[0]: hp — homogeneous landmark [x, y, z, w] (4D)
//! ```
//!
//! # Jacobian (3×4)
//!
//! ```text
//! ∂e/∂hp = −sqrt_info · [I₃ | −p_est] / w
//! ```

use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3};

use crate::factors::Factor;

/// Unary factor constraining a homogeneous 4D point to a measured 3D position.
pub struct HomogeneousPointFactor {
    /// Measured 3D Euclidean position.
    measurement: Vector3<f64>,
    /// Square-root information matrix (3×3, `W` s.t. `Wᵀ W = Σ⁻¹`).
    sqrt_information: SMatrix<f64, 3, 3>,
}

impl HomogeneousPointFactor {
    /// Create a homogeneous point factor.
    ///
    /// # Arguments
    /// * `measurement` — measured 3D point position
    /// * `sqrt_information` — 3×3 square-root information matrix
    pub fn new(measurement: Vector3<f64>, sqrt_information: SMatrix<f64, 3, 3>) -> Self {
        Self {
            measurement,
            sqrt_information,
        }
    }

    /// Create with isotropic noise (scalar standard deviation).
    pub fn new_isotropic(measurement: Vector3<f64>, sigma: f64) -> Self {
        let sqrt_info = SMatrix::<f64, 3, 3>::identity() * (1.0 / sigma);
        Self::new(measurement, sqrt_info)
    }
}

impl Factor for HomogeneousPointFactor {
    fn get_dimension(&self) -> usize {
        3
    }

    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        debug_assert_eq!(
            params.len(),
            1,
            "HomogeneousPointFactor expects 1 parameter block"
        );
        debug_assert_eq!(
            params[0].len(),
            4,
            "params[0] must be homogeneous point (4D)"
        );

        let hp = &params[0];
        let w = hp[3];

        // Dehomogenize
        let p_est = Vector3::new(hp[0] / w, hp[1] / w, hp[2] / w);

        // Unweighted residual
        let raw_err = self.measurement - p_est;

        // Weighted residual
        let weighted_err = self.sqrt_information * raw_err;
        let residual = DVector::from_iterator(3, weighted_err.iter().copied());

        if !compute_jacobian {
            return (residual, None);
        }

        // ── Jacobian ──────────────────────────────────────────────────────────
        //
        // p_est = hp[0:3] / w
        // e = p_meas - p_est
        //
        // ∂p_est/∂hp = [1/w  0    0   -x/w²]   = [I₃ | -p_est] / w
        //              [0    1/w  0   -y/w²]
        //              [0    0    1/w -z/w²]
        //
        // ∂e/∂hp = -∂p_est/∂hp = -[I₃ | -p_est] / w

        let inv_w = 1.0 / w;
        let mut j_raw = SMatrix::<f64, 3, 4>::zeros();
        // First 3 cols: -I₃/w
        j_raw
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-Matrix3::identity() * inv_w));
        // Last col: p_est/w = hp[0:3]/w²
        j_raw[(0, 3)] = p_est[0] * inv_w;
        j_raw[(1, 3)] = p_est[1] * inv_w;
        j_raw[(2, 3)] = p_est[2] * inv_w;

        let j_weighted = self.sqrt_information * j_raw;
        let jac = DMatrix::from_iterator(3, 4, j_weighted.iter().copied());

        (residual, Some(jac))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    // ── Test 1: zero residual when measurement matches dehomogenized point ──

    #[test]
    fn zero_residual_at_measurement() {
        let hp = dvector![2.0, 4.0, 6.0, 2.0]; // p_est = [1, 2, 3]
        let measurement = Vector3::new(1.0, 2.0, 3.0);

        let factor = HomogeneousPointFactor::new_isotropic(measurement, 1.0);
        let (r, _) = factor.linearize(&[hp], false);

        for i in 0..3 {
            assert!(
                r[i].abs() < 1e-12,
                "residual[{i}] = {} should be zero",
                r[i]
            );
        }
    }

    // ── Test 2: zero residual with w != 1 ───────────────────────────────────

    #[test]
    fn zero_residual_with_nonunit_w() {
        let w = 0.5;
        let p = Vector3::new(3.0, -1.0, 7.0);
        let hp = dvector![p[0] * w, p[1] * w, p[2] * w, w];

        let factor = HomogeneousPointFactor::new_isotropic(p, 0.5);
        let (r, _) = factor.linearize(&[hp], false);

        for i in 0..3 {
            assert!(r[i].abs() < 1e-12, "residual[{i}] = {}", r[i]);
        }
    }

    // ── Test 3: non-zero residual ───────────────────────────────────────────

    #[test]
    fn nonzero_residual() {
        let hp = dvector![1.0, 2.0, 3.0, 1.0];
        let measurement = Vector3::new(2.0, 2.0, 3.0); // differs in x by 1

        let factor = HomogeneousPointFactor::new_isotropic(measurement, 1.0);
        let (r, _) = factor.linearize(&[hp], false);

        assert!((r[0] - 1.0).abs() < 1e-12);
        assert!(r[1].abs() < 1e-12);
        assert!(r[2].abs() < 1e-12);
    }

    // ── Test 4: finite-difference Jacobian check ────────────────────────────

    #[test]
    fn finite_difference_jacobian() {
        let hp = dvector![2.0, -1.5, 4.0, 0.8];
        let measurement = Vector3::new(1.0, 2.0, 3.0);

        let sqrt_info = SMatrix::<f64, 3, 3>::new(1.0, 0.1, 0.0, 0.0, 2.0, 0.05, 0.0, 0.0, 0.5);
        let factor = HomogeneousPointFactor::new(measurement, sqrt_info);

        let (r0, jac_opt) = factor.linearize(&[hp.clone()], true);
        let jac = jac_opt.expect("Jacobian must be computed");

        const EPS: f64 = 1e-7;
        const TOL: f64 = 1e-5;

        for col in 0..4 {
            let mut hp_pert = hp.clone();
            hp_pert[col] += EPS;
            let (r_pert, _) = factor.linearize(&[hp_pert], false);
            let fd = (&r_pert - &r0) / EPS;
            for row in 0..3 {
                let err = (fd[row] - jac[(row, col)]).abs();
                assert!(
                    err < TOL,
                    "J[{row},{col}]: analytical={:.8} fd={:.8} err={err:.2e}",
                    jac[(row, col)],
                    fd[row]
                );
            }
        }
    }

    // ── Test 5: dimension ───────────────────────────────────────────────────

    #[test]
    fn dimension_is_three() {
        let factor = HomogeneousPointFactor::new_isotropic(Vector3::zeros(), 1.0);
        assert_eq!(factor.get_dimension(), 3);
    }
}
