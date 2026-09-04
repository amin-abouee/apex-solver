//! Numerical utilities shared by factor implementations.

use nalgebra::{DMatrix, Matrix3, SMatrix, Vector3};

/// Numerically stable sinc: `sinc(x) = sin(x) / x`.
///
/// Uses a Taylor expansion for |x|² < 1e-10 to avoid division by near-zero.
pub fn sinc(x: f64) -> f64 {
    let x2 = x * x;
    if x2 < 1e-10 {
        1.0 - x2 / 6.0 + x2 * x2 / 120.0
    } else {
        x.sin() / x
    }
}

/// Skew-symmetric (cross-product) matrix: `[v]× w = v × w`.
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y, //
        v.z, 0.0, -v.x, //
        -v.y, v.x, 0.0,
    )
}

/// Symmetric pseudo-inverse square root of a `D`×`D` PSD matrix.
///
/// Returns `U` such that `Uᵀ · U ≈ M⁻¹`. Eigenvalues below `1e-12` are clamped
/// to zero, so a rank-deficient `M` yields the pseudo-inverse root rather than
/// non-finite entries.
///
/// The eigendecomposition runs through a heap `DMatrix` rather than the
/// const-generic `SMatrix` path: expressing `SymmetricEigen`'s allocator bounds
/// generically over `Const<D>` does not normalize back to `ArrayStorage`, and
/// the alternative is one hand-written copy per dimension. The public signature
/// stays fixed-size, and the single small allocation is paid once per
/// preintegration rebuild — never inside `linearize`.
pub fn symm_sqrt_inverse<const D: usize>(m: &SMatrix<f64, D, D>) -> SMatrix<f64, D, D> {
    let eigen = nalgebra::SymmetricEigen::new(DMatrix::from_iterator(D, D, m.iter().copied()));
    let epsilon = 1e-12;

    let mut s_inv_sqrt = SMatrix::<f64, D, D>::zeros();
    for i in 0..D {
        let ev = eigen.eigenvalues[i];
        if ev > epsilon {
            s_inv_sqrt[(i, i)] = 1.0 / ev.sqrt();
        }
    }
    // U = S^{-1/2} · Vᵀ,  so that Uᵀ U = V S⁻¹ Vᵀ = M⁻¹
    let eigenvectors = SMatrix::<f64, D, D>::from_iterator(eigen.eigenvectors.iter().copied());
    s_inv_sqrt * eigenvectors.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinc_at_zero() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn sinc_small_angle_matches_exact() {
        let x = 1e-8_f64;
        assert!((sinc(x) - x.sin() / x).abs() < 1e-14);
    }

    #[test]
    fn sinc_large_angle() {
        let x = 1.0_f64;
        assert!((sinc(x) - x.sin() / x).abs() < 1e-15);
    }

    #[test]
    fn skew_matches_cross_product() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let w = Vector3::new(4.0, 5.0, 6.0);
        assert!((skew(&v) * w - v.cross(&w)).norm() < 1e-14);
    }

    #[test]
    fn symm_sqrt_inverse_of_identity() {
        let id = SMatrix::<f64, 15, 15>::identity();
        assert!((symm_sqrt_inverse(&id) - id).norm() < 1e-10);
    }

    #[test]
    fn symm_sqrt_inverse_roundtrip() {
        let mut m = SMatrix::<f64, 15, 15>::identity() * 2.0;
        m[(0, 1)] = 0.5;
        m[(1, 0)] = 0.5;
        let u = symm_sqrt_inverse(&m);
        let product = u.transpose() * u * m;
        let id = SMatrix::<f64, 15, 15>::identity();
        assert!((product - id).norm() < 1e-8);
    }

    #[test]
    fn symm_sqrt_inverse_generic_over_dimension() {
        // The 9×9 instantiation the non-combined IMU factors rely on.
        let mut m = SMatrix::<f64, 9, 9>::identity() * 4.0;
        m[(2, 5)] = 0.25;
        m[(5, 2)] = 0.25;
        let u = symm_sqrt_inverse(&m);
        let product = u.transpose() * u * m;
        assert!((product - SMatrix::<f64, 9, 9>::identity()).norm() < 1e-8);
    }

    #[test]
    fn symm_sqrt_inverse_clamps_rank_deficient_directions() {
        // A singular matrix must not produce non-finite entries.
        let mut m = SMatrix::<f64, 3, 3>::identity();
        m[(2, 2)] = 0.0;
        let u = symm_sqrt_inverse(&m);
        assert!(u.iter().all(|v| v.is_finite()));
    }
}
