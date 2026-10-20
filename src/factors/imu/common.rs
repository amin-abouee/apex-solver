//! Block plumbing shared by the group-specific IMU factors.

use faer::prelude::ReborrowMut;
use nalgebra::{SMatrix, Vector3};

/// Read a 6D bias block as `(b_g, b_a)`.
pub(super) fn split_bias(block: &[f64]) -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(block[0], block[1], block[2]),
        Vector3::new(block[3], block[4], block[5]),
    )
}

/// Copy a weighted Jacobian into the caller's column-major buffer.
pub(super) fn write_jacobian<const R: usize, const C: usize>(
    weighted: &SMatrix<f64, R, C>,
    jac: &mut faer::mat::MatMut<'_, f64>,
) {
    for row in 0..R {
        for col in 0..C {
            *jac.rb_mut().get_mut(row, col) = weighted[(row, col)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_block_splits_into_gyro_and_accel() {
        let (gyro, accel) = split_bias(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(gyro, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(accel, Vector3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn jacobian_lands_column_major() {
        let m = SMatrix::<f64, 2, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let mut buf = vec![0.0; 6];
        let mut view = faer::mat::MatMut::from_column_major_slice_mut(&mut buf, 2, 3);
        write_jacobian(&m, &mut view);
        assert_eq!(buf, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
