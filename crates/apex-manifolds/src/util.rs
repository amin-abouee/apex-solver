//! Small shared helpers used by the group implementations and by their consumers.

use nalgebra::{Matrix3, Vector3};

/// Skew-symmetric (cross-product) matrix: `[v]× w = v × w`.
#[inline]
#[must_use]
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skew_matches_the_cross_product() {
        let v = Vector3::new(1.0, -2.0, 3.0);
        let w = Vector3::new(-0.5, 4.0, 0.25);
        assert_eq!(skew(&v) * w, v.cross(&w));
    }

    #[test]
    fn skew_is_antisymmetric() {
        let m = skew(&Vector3::new(0.3, 0.7, -1.1));
        assert_eq!(m + m.transpose(), Matrix3::zeros());
    }
}
