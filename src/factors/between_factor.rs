use super::Factor;
use apex_manifolds::{LieGroup, Tangent};
use faer::prelude::ReborrowMut;

/// Generic between factor for Lie group pose constraints.
///
/// Represents a relative pose measurement between two poses of any Lie group manifold type.
/// This is a generic implementation that works with SE(2), SE(3), SO(2), SO(3), and Rⁿ
/// using static dispatch for zero runtime overhead.
///
/// # Type Parameter
///
/// * `T` - The Lie group manifold type (e.g., SE2, SE3, SO2, SO3, Rn)
///
/// # Mathematical Formulation
///
/// Given two poses `T_i` and `T_j` in a Lie group, and a measurement `T_ij`, the residual is:
///
/// ```text
/// r = log(T_ij⁻¹ ⊕ T_i⁻¹ ⊕ T_j)
/// ```
///
/// where:
/// - `⊕` is the Lie group composition operation
/// - `log` is the logarithm map (converts from manifold to tangent space)
/// - The residual dimensionality depends on the manifold's degrees of freedom (DOF)
///
/// # Residual Dimensions by Manifold Type
///
/// - **SE(3)**: 6D residual `[v_x, v_y, v_z, ω_x, ω_y, ω_z]` - translation + rotation
/// - **SE(2)**: 3D residual `[dx, dy, dθ]` - 2D translation + rotation
/// - **SO(3)**: 3D residual `[ω_x, ω_y, ω_z]` - 3D rotation only
/// - **SO(2)**: 1D residual `[dθ]` - 2D rotation only
/// - **Rⁿ**: nD residual - Euclidean space
///
/// # Jacobian Computation
///
/// The Jacobian is computed analytically using the chain rule and Lie group derivatives:
///
/// ```text
/// J = ∂r/∂[T_i, T_j]
/// ```
///
/// The Jacobian dimensions are `DOF × (2 × DOF)` where DOF is the manifold's degrees of freedom:
/// - **SE(3)**: 6×12 matrix
/// - **SE(2)**: 3×6 matrix
/// - **SO(3)**: 3×6 matrix
/// - **SO(2)**: 1×2 matrix
///
/// # Use Cases
///
/// - **3D SLAM**: Visual odometry, loop closure constraints (SE3)
/// - **2D SLAM**: Robot navigation, mapping (SE2)
/// - **Pose graph optimization**: Relative pose constraints (SE2, SE3)
/// - **Orientation tracking**: IMU fusion, attitude estimation (SO2, SO3)
/// - **General manifold optimization**: Custom manifolds (Rⁿ)
///
/// # Examples
///
/// ## SE(3) - 3D Pose Graph
///
/// ```
/// use apex_solver::factors::{Factor, BetweenFactor};
/// use apex_solver::manifold::se3::SE3;
/// use nalgebra::{Vector3, Quaternion, DVector};
///
/// let relative_pose = SE3::from_translation_quaternion(
///     Vector3::new(1.0, 0.0, 0.0),
///     Quaternion::new(1.0, 0.0, 0.0, 0.0),
/// );
/// let between = BetweenFactor::new(relative_pose);
///
/// let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
/// let pose_j = DVector::from_vec(vec![0.95, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0]);
///
/// let mut residual = vec![0.0f64; between.residual_dim()];
/// let (rows, cols) = between.jacobian_shape();
/// let mut jac_buf = vec![0.0f64; rows * cols];
/// let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
/// between.linearize(&[pose_i.as_slice(), pose_j.as_slice()], &mut residual, Some(jac_mut));
/// ```
///
/// ## SE(2) - 2D Pose Graph
///
/// ```
/// use apex_solver::factors::{Factor, BetweenFactor};
/// use apex_solver::manifold::se2::SE2;
/// use nalgebra::DVector;
///
/// let relative_pose = SE2::from_xy_angle(1.0, 0.0, 0.1);
/// let between = BetweenFactor::new(relative_pose);
///
/// let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0]);
/// let pose_j = DVector::from_vec(vec![0.95, 0.05, 0.12]);
///
/// let mut residual = vec![0.0f64; between.residual_dim()];
/// between.linearize(&[pose_i.as_slice(), pose_j.as_slice()], &mut residual, None);
/// ```
///
/// # Performance
///
/// This generic implementation uses static dispatch (monomorphization), meaning:
/// - **Zero runtime overhead** compared to type-specific implementations
/// - Compiler optimizes each instantiation (`BetweenFactor<SE3>`, `BetweenFactor<SE2>`, etc.)
/// - All type checking happens at compile time
/// - No dynamic dispatch or virtual function calls
#[derive(Clone, PartialEq)]
pub struct BetweenFactor<T>
where
    T: LieGroup + Clone + Send + Sync,
{
    /// The measured relative pose transformation between the two connected poses
    pub relative_pose: T,
}

impl<T> BetweenFactor<T>
where
    T: LieGroup + Clone + Send + Sync,
{
    /// Create a new between factor from a relative pose measurement.
    ///
    /// This is a generic constructor that works with any Lie group manifold type.
    /// The type parameter `T` is typically inferred from the `relative_pose` argument.
    ///
    /// # Arguments
    ///
    /// * `relative_pose` - The measured relative transformation between two poses
    ///
    /// # Returns
    ///
    /// A new `BetweenFactor<T>` instance
    ///
    /// # Examples
    ///
    /// ## SE(3) Between Factor
    ///
    /// ```
    /// use apex_solver::factors::BetweenFactor;
    /// use apex_solver::manifold::se3::SE3;
    ///
    /// // Create relative pose: move 2m in x, rotate 90° around z-axis
    /// let relative = SE3::from_translation_euler(
    ///     2.0, 0.0, 0.0,                      // translation (x, y, z)
    ///     0.0, 0.0, std::f64::consts::FRAC_PI_2  // rotation (roll, pitch, yaw)
    /// );
    ///
    /// // Type is inferred as BetweenFactor<SE3>
    /// let factor = BetweenFactor::new(relative);
    /// ```
    ///
    /// ## SE(2) Between Factor
    ///
    /// ```
    /// use apex_solver::factors::BetweenFactor;
    /// use apex_solver::manifold::se2::SE2;
    ///
    /// // Create relative 2D pose
    /// let relative = SE2::from_xy_angle(1.0, 0.5, 0.1);
    ///
    /// // Type is inferred as BetweenFactor<SE2>
    /// let factor = BetweenFactor::new(relative);
    /// ```
    pub fn new(relative_pose: T) -> Self {
        Self { relative_pose }
    }
}

impl<T> Factor for BetweenFactor<T>
where
    T: LieGroup + Clone + Send + Sync,
{
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let se3_origin_k0 = T::from_param_slice(params[0]);
        let se3_origin_k1 = T::from_param_slice(params[1]);
        let se3_k0_k1_measured = &self.relative_pose;

        // Step 1: se3_origin_k1.between(se3_origin_k0) = k1⁻¹ * k0
        let mut j_k1_k0_wrt_k1 = T::zero_jacobian();
        let mut j_k1_k0_wrt_k0 = T::zero_jacobian();
        let se3_k1_k0 = se3_origin_k1.between(
            &se3_origin_k0,
            Some(&mut j_k1_k0_wrt_k1),
            Some(&mut j_k1_k0_wrt_k0),
        );

        // Step 2: se3_k1_k0 * se3_k0_k1_measured
        let mut j_diff_wrt_k1_k0 = T::zero_jacobian();
        let se3_diff = se3_k1_k0.compose(se3_k0_k1_measured, Some(&mut j_diff_wrt_k1_k0), None);

        // Step 3: se3_diff.log()
        let mut j_log_wrt_diff = T::zero_jacobian();
        let tangent = se3_diff.log(Some(&mut j_log_wrt_diff));
        let tangent_slice = tangent.as_slice();
        let dof = tangent_slice.len();

        residual[..dof].copy_from_slice(tangent_slice);

        if let Some(mut jac) = jacobian {
            let j_diff_wrt_k0 = j_diff_wrt_k1_k0.clone() * j_k1_k0_wrt_k0;
            let j_diff_wrt_k1 = j_diff_wrt_k1_k0 * j_k1_k0_wrt_k1;
            let jacobian_wrt_k0 = j_log_wrt_diff.clone() * j_diff_wrt_k0;
            let jacobian_wrt_k1 = j_log_wrt_diff * j_diff_wrt_k1;

            for i in 0..dof {
                for j in 0..dof {
                    *jac.rb_mut().get_mut(i, j) = jacobian_wrt_k0[(i, j)];
                    *jac.rb_mut().get_mut(i, j + dof) = jacobian_wrt_k1[(i, j)];
                }
            }
        }
    }

    fn residual_dim(&self) -> usize {
        self.relative_pose.tangent_dim()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        let dof = self.relative_pose.tangent_dim();
        (dof, 2 * dof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::rn::Rn;
    use apex_manifolds::se2::{SE2, SE2Tangent};
    use apex_manifolds::se3::SE3;
    use apex_manifolds::so2::SO2;
    use apex_manifolds::so3::SO3;
    use nalgebra::{DMatrix, DVector, Quaternion, Vector3};

    const TOLERANCE: f64 = 1e-9;
    const FD_EPSILON: f64 = 1e-6;
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn compute_residual<T>(
        factor: &BetweenFactor<T>,
        pose_i: &DVector<f64>,
        pose_j: &DVector<f64>,
    ) -> Vec<f64>
    where
        T: LieGroup + Clone + Send + Sync,
    {
        let mut residual = vec![0.0f64; factor.residual_dim()];
        factor.linearize(&[pose_i.as_slice(), pose_j.as_slice()], &mut residual, None);
        residual
    }

    fn compute_with_jacobian<T>(
        factor: &BetweenFactor<T>,
        pose_i: &DVector<f64>,
        pose_j: &DVector<f64>,
    ) -> (Vec<f64>, DMatrix<f64>)
    where
        T: LieGroup + Clone + Send + Sync,
    {
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0f64; rows];
        let mut jac_buf = vec![0.0f64; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(
            &[pose_i.as_slice(), pose_j.as_slice()],
            &mut residual,
            Some(jac_mut),
        );
        let jacobian = DMatrix::from_column_slice(rows, cols, &jac_buf);
        (residual, jacobian)
    }

    /// `BetweenFactor<Rn>` is a documented supported combination, but `Rn` is
    /// dynamically sized, so its Jacobian blocks must follow the element's
    /// runtime dimension rather than a hardcoded 3×3.
    ///
    /// Regression for the dimension-mismatch panic inside `linearize`.
    #[test]
    fn test_between_factor_rn_jacobian_matches_dimension() -> TestResult {
        for dim in [1usize, 2, 3, 4, 7] {
            let relative = Rn::from_slice(&vec![0.25f64; dim]);
            let factor = BetweenFactor::new(relative);

            let pose_i = DVector::from_vec(vec![0.0f64; dim]);
            let pose_j = DVector::from_vec(vec![1.0f64; dim]);

            let (rows, cols) = factor.jacobian_shape();
            assert_eq!(rows, dim, "residual rows for Rn({dim})");
            assert_eq!(cols, 2 * dim, "jacobian cols for Rn({dim})");

            let (residual, jacobian) = compute_with_jacobian(&factor, &pose_i, &pose_j);
            assert_eq!(residual.len(), dim);
            assert_eq!(jacobian.nrows(), dim);
            assert_eq!(jacobian.ncols(), 2 * dim);
            assert!(
                jacobian.iter().all(|v| v.is_finite()),
                "non-finite Jacobian entry for Rn({dim})"
            );
        }
        Ok(())
    }

    /// The Rⁿ between-factor residual is `(x_i - x_j) + measurement`, and the
    /// Jacobian blocks are ∓I. Verified against finite differences so the
    /// dimension fix cannot silently produce a wrong-but-well-shaped block.
    #[test]
    fn test_between_factor_rn_jacobian_matches_finite_differences() -> TestResult {
        for dim in [2usize, 4] {
            let relative = Rn::from_slice(&vec![0.5f64; dim]);
            let factor = BetweenFactor::new(relative);

            let pose_i = DVector::from_vec((0..dim).map(|k| 0.1 * k as f64).collect::<Vec<_>>());
            let pose_j = DVector::from_vec((0..dim).map(|k| 1.0 - 0.2 * k as f64).collect::<Vec<_>>());

            let (_, analytic) = compute_with_jacobian(&factor, &pose_i, &pose_j);

            for (block, pose) in [(0usize, &pose_i), (1usize, &pose_j)] {
                for c in 0..dim {
                    let mut plus = pose.clone();
                    let mut minus = pose.clone();
                    plus[c] += FD_EPSILON;
                    minus[c] -= FD_EPSILON;

                    let (rp, rm) = if block == 0 {
                        (
                            compute_residual(&factor, &plus, &pose_j),
                            compute_residual(&factor, &minus, &pose_j),
                        )
                    } else {
                        (
                            compute_residual(&factor, &pose_i, &plus),
                            compute_residual(&factor, &pose_i, &minus),
                        )
                    };

                    for r in 0..dim {
                        let fd = (rp[r] - rm[r]) / (2.0 * FD_EPSILON);
                        let an = analytic[(r, block * dim + c)];
                        assert!(
                            (fd - an).abs() < 1e-6,
                            "Rn({dim}) block {block} entry ({r},{c}): analytic {an}, fd {fd}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_between_factor_se2_identity() {
        let relative = SE2::identity();
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![0.0, 0.0, 0.0]);

        let residual = compute_residual(&factor, &pose_i, &pose_j);

        assert_eq!(residual.len(), 3);
        let norm: f64 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < TOLERANCE, "Residual norm: {}", norm);
    }

    #[test]
    fn test_between_factor_se3_identity() {
        let relative = SE3::identity();
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

        let residual = compute_residual(&factor, &pose_i, &pose_j);

        assert_eq!(residual.len(), 6);
        let norm: f64 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < TOLERANCE, "Residual norm: {}", norm);
    }

    #[test]
    fn test_between_factor_se2_jacobian_numerical() -> TestResult {
        let relative = SE2::from_xy_angle(1.0, 0.0, 0.1);
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![0.95, 0.05, 0.12]);

        let (residual, jacobian) = compute_with_jacobian(&factor, &pose_i, &pose_j);

        assert_eq!(jacobian.nrows(), 3);
        assert_eq!(jacobian.ncols(), 6);

        let mut jacobian_fd = DMatrix::<f64>::zeros(3, 6);
        let se2_i = SE2::from_param_slice(pose_i.as_slice());
        let se2_j = SE2::from_param_slice(pose_j.as_slice());

        for i in 0..3 {
            let delta = match i {
                0 => SE2Tangent::new(FD_EPSILON, 0.0, 0.0),
                1 => SE2Tangent::new(0.0, FD_EPSILON, 0.0),
                2 => SE2Tangent::new(0.0, 0.0, FD_EPSILON),
                _ => unreachable!(),
            };
            let pose_i_p =
                DVector::from_column_slice(se2_i.plus(&delta, None, None).as_param_slice());
            let residual_p = compute_residual(&factor, &pose_i_p, &pose_j);
            for j in 0..3 {
                jacobian_fd[(j, i)] = (residual_p[j] - residual[j]) / FD_EPSILON;
            }
        }

        for i in 0..3 {
            let delta = match i {
                0 => SE2Tangent::new(FD_EPSILON, 0.0, 0.0),
                1 => SE2Tangent::new(0.0, FD_EPSILON, 0.0),
                2 => SE2Tangent::new(0.0, 0.0, FD_EPSILON),
                _ => unreachable!(),
            };
            let pose_j_p =
                DVector::from_column_slice(se2_j.plus(&delta, None, None).as_param_slice());
            let residual_p = compute_residual(&factor, &pose_i, &pose_j_p);
            for j in 0..3 {
                jacobian_fd[(j, i + 3)] = (residual_p[j] - residual[j]) / FD_EPSILON;
            }
        }

        let diff_norm = (jacobian - jacobian_fd).norm();
        assert!(diff_norm < 1e-5, "Jacobian difference norm: {}", diff_norm);
        Ok(())
    }

    #[test]
    fn test_between_factor_se3_jacobian_numerical() -> TestResult {
        let relative = SE3::from_translation_quaternion(
            Vector3::new(1.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
        );
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![0.95, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0]);

        let (residual, jacobian) = compute_with_jacobian(&factor, &pose_i, &pose_j);

        assert_eq!(jacobian.nrows(), 6);
        assert_eq!(jacobian.ncols(), 12);

        let mut jacobian_fd = DMatrix::<f64>::zeros(6, 12);

        for i in 0..3 {
            let mut pose_i_p = pose_i.clone();
            pose_i_p[i] += FD_EPSILON;
            let residual_p = compute_residual(&factor, &pose_i_p, &pose_j);
            for j in 0..6 {
                jacobian_fd[(j, i)] = (residual_p[j] - residual[j]) / FD_EPSILON;
            }
        }

        for i in 0..3 {
            let mut pose_j_p = pose_j.clone();
            pose_j_p[i] += FD_EPSILON;
            let residual_p = compute_residual(&factor, &pose_i, &pose_j_p);
            for j in 0..6 {
                jacobian_fd[(j, i + 6)] = (residual_p[j] - residual[j]) / FD_EPSILON;
            }
        }

        let diff_norm_trans = (jacobian.columns(0, 3) - jacobian_fd.columns(0, 3)).norm();
        assert!(
            diff_norm_trans < 1e-5,
            "Jacobian difference norm (translation): {}",
            diff_norm_trans
        );
        Ok(())
    }

    #[test]
    fn test_between_factor_dimension_se2() -> TestResult {
        let relative = SE2::from_xy_angle(1.0, 0.5, 0.1);
        let factor = BetweenFactor::new(relative);

        assert_eq!(factor.residual_dim(), 3);
        assert_eq!(factor.jacobian_shape(), (3, 6));

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let (residual, jacobian) = compute_with_jacobian(&factor, &pose_i, &pose_j);

        assert_eq!(residual.len(), 3);
        assert_eq!(jacobian.nrows(), 3);
        assert_eq!(jacobian.ncols(), 6);
        Ok(())
    }

    #[test]
    fn test_between_factor_dimension_se3() -> TestResult {
        let relative = SE3::identity();
        let factor = BetweenFactor::new(relative);

        assert_eq!(factor.residual_dim(), 6);
        assert_eq!(factor.jacobian_shape(), (6, 12));

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let (residual, jacobian) = compute_with_jacobian(&factor, &pose_i, &pose_j);

        assert_eq!(residual.len(), 6);
        assert_eq!(jacobian.nrows(), 6);
        assert_eq!(jacobian.ncols(), 12);
        Ok(())
    }

    #[test]
    fn test_between_factor_so2_so3() -> TestResult {
        let so2_relative = SO2::from_angle(0.1);
        let so2_factor = BetweenFactor::new(so2_relative);

        assert_eq!(so2_factor.residual_dim(), 1);
        assert_eq!(so2_factor.jacobian_shape(), (1, 2));

        let so2_i = DVector::from_vec(vec![0.0]);
        let so2_j = DVector::from_vec(vec![0.12]);
        let (res_so2, jac_so2) = compute_with_jacobian(&so2_factor, &so2_i, &so2_j);
        assert_eq!(res_so2.len(), 1);
        assert_eq!(jac_so2.nrows(), 1);
        assert_eq!(jac_so2.ncols(), 2);

        let so3_relative = SO3::identity();
        let so3_factor = BetweenFactor::new(so3_relative);

        let so3_i = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let so3_j = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let (res_so3, jac_so3) = compute_with_jacobian(&so3_factor, &so3_i, &so3_j);
        assert_eq!(res_so3.len(), 3);
        assert_eq!(jac_so3.nrows(), 3);
        assert_eq!(jac_so3.ncols(), 6);
        Ok(())
    }

    #[test]
    fn test_between_factor_finiteness() -> TestResult {
        let relative = SE2::from_xy_angle(100.0, -200.0, std::f64::consts::PI);
        let factor = BetweenFactor::new(relative);

        let pose_i = DVector::from_vec(vec![50.0, -100.0, 1.5]);
        let pose_j = DVector::from_vec(vec![150.0, -300.0, -1.5]);

        let (residual, jacobian) = compute_with_jacobian(&factor, &pose_i, &pose_j);

        assert!(residual.iter().all(|x| x.is_finite()));
        assert!(jacobian.iter().all(|x| x.is_finite()));
        Ok(())
    }

    #[test]
    fn test_between_factor_clone() {
        let relative = SE3::identity();
        let factor = BetweenFactor::new(relative);
        let factor_clone = factor.clone();

        let pose_i = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let pose_j = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

        let r1 = compute_residual(&factor, &pose_i, &pose_j);
        let r2 = compute_residual(&factor_clone, &pose_i, &pose_j);

        let diff: f64 = r1.iter().zip(r2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < TOLERANCE);
    }
}
