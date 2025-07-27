//! Comprehensive unit tests for factor implementations
//!
//! This module contains unit tests for all factor types, testing mathematical
//! correctness, Jacobian computations, and integration with manifold operations.

#[cfg(test)]
mod tests {
    use crate::manifold::{se2::SE2, se3::SE3, so2::SO2, so3::SO3, LieGroup};
    use nalgebra::{DMatrix, DVector, Vector2, Vector3, Matrix2};

    // Test utilities
    fn create_test_se2() -> SE2 {
        SE2::from_translation_and_angle(Vector2::new(1.0, 2.0), 0.5)
    }

    fn create_test_se3() -> SE3 {
        SE3::from_translation_and_rotation(
            Vector3::new(1.0, 2.0, 3.0),
            SO3::from_euler_angles(0.1, 0.2, 0.3),
        )
    }

    fn create_test_so2() -> SO2 {
        SO2::from_angle(0.5)
    }

    fn create_test_so3() -> SO3 {
        SO3::from_euler_angles(0.1, 0.2, 0.3)
    }

    // Basic factor tests
    mod basic_tests {
        use super::*;
        use crate::factors::basic::*;

        #[test]
        fn test_unary_factor_creation() {
            let measurement = create_test_se3();
            let information = DMatrix::identity(6, 6);
            let factor = UnaryFactor::new(1, 100, measurement.clone(), information.clone());

            assert_eq!(factor.id(), 1);
            assert_eq!(factor.variable_keys(), &[100]);
            assert_eq!(factor.measurement(), &measurement);
            assert_eq!(factor.information(), &information);
        }

        #[test]
        fn test_binary_factor_creation() {
            let measurement = create_test_se3();
            let information = DMatrix::identity(6, 6);
            let factor = BinaryFactor::new(2, [100, 101], measurement.clone(), information.clone());

            assert_eq!(factor.id(), 2);
            assert_eq!(factor.variable_keys(), &[100, 101]);
            assert_eq!(factor.measurement(), &measurement);
            assert_eq!(factor.information(), &information);
        }

        #[test]
        fn test_prior_factor_identity() {
            let prior_value = create_test_se2();
            let factor = PriorFactor::new_identity(3, 200, prior_value.clone());

            assert_eq!(factor.id(), 3);
            assert_eq!(factor.variable_keys(), &[200]);
            assert_eq!(factor.measurement(), &prior_value);
            
            let expected_info = DMatrix::identity(3, 3);
            assert_eq!(factor.information(), &expected_info);
        }

        #[test]
        fn test_prior_factor_diagonal() {
            let prior_value = create_test_se3();
            let diagonal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let factor = PriorFactor::new_diagonal(4, 300, prior_value, &diagonal).unwrap();

            let expected_info = DMatrix::from_diagonal(&DVector::from_vec(diagonal.to_vec()));
            assert_eq!(factor.information(), &expected_info);
        }

        #[test]
        fn test_prior_factor_diagonal_wrong_size() {
            let prior_value = create_test_se3();
            let diagonal = [1.0, 2.0, 3.0]; // Wrong size for SE3 (should be 6)
            let result = PriorFactor::new_diagonal(5, 400, prior_value, &diagonal);
            
            assert!(result.is_err());
        }
    }

    // Geometry factor tests
    mod geometry_tests {
        use super::*;
        use crate::factors::geometry::*;

        #[test]
        fn test_between_factor_creation() {
            let measurement = create_test_se3();
            let information = DMatrix::identity(6, 6);
            let factor = BetweenFactor::new(10, [500, 501], measurement.clone(), information.clone());

            assert_eq!(factor.id(), 10);
            assert_eq!(factor.variable_keys(), &[500, 501]);
            assert_eq!(factor.pose_keys(), [500, 501]);
            assert_eq!(factor.measurement(), &measurement);
            assert_eq!(factor.information(), &information);
        }

        #[test]
        fn test_se2_between_factor_identity() {
            let measurement = create_test_se2();
            let factor = SE2BetweenFactor::new_identity(11, [600, 601], measurement);

            let expected_info = DMatrix::identity(3, 3);
            assert_eq!(factor.information(), &expected_info);
        }

        #[test]
        fn test_se2_between_factor_diagonal() {
            let measurement = create_test_se2();
            let trans_prec = 10.0;
            let rot_prec = 5.0;
            let factor = SE2BetweenFactor::new_diagonal(12, [700, 701], measurement, trans_prec, rot_prec);

            let info = factor.information();
            assert_eq!(info[(0, 0)], trans_prec);
            assert_eq!(info[(1, 1)], trans_prec);
            assert_eq!(info[(2, 2)], rot_prec);
        }

        #[test]
        fn test_se3_between_factor_identity() {
            let measurement = create_test_se3();
            let factor = SE3BetweenFactor::new_identity(13, [800, 801], measurement);

            let expected_info = DMatrix::identity(6, 6);
            assert_eq!(factor.information(), &expected_info);
        }

        #[test]
        fn test_se3_between_factor_diagonal() {
            let measurement = create_test_se3();
            let trans_prec = 10.0;
            let rot_prec = 5.0;
            let factor = SE3BetweenFactor::new_diagonal(14, [900, 901], measurement, trans_prec, rot_prec);

            let info = factor.information();
            for i in 0..3 {
                assert_eq!(info[(i, i)], trans_prec);
            }
            for i in 3..6 {
                assert_eq!(info[(i, i)], rot_prec);
            }
        }

        #[test]
        fn test_se3_between_factor_split_precision() {
            let measurement = create_test_se3();
            let trans_prec = [1.0, 2.0, 3.0];
            let rot_prec = [4.0, 5.0, 6.0];
            let factor = SE3BetweenFactor::new_split_precision(15, [1000, 1001], measurement, &trans_prec, &rot_prec);

            let info = factor.information();
            for i in 0..3 {
                assert_eq!(info[(i, i)], trans_prec[i]);
            }
            for i in 0..3 {
                assert_eq!(info[(i + 3, i + 3)], rot_prec[i]);
            }
        }

        #[test]
        fn test_so2_between_factor_scalar() {
            let measurement = create_test_so2();
            let precision = 2.5;
            let factor = SO2BetweenFactor::new_scalar(16, [1100, 1101], measurement, precision);

            let info = factor.information();
            assert_eq!(info.nrows(), 1);
            assert_eq!(info.ncols(), 1);
            assert_eq!(info[(0, 0)], precision);
        }

        #[test]
        fn test_so3_between_factor_axis_precision() {
            let measurement = create_test_so3();
            let axis_prec = [1.0, 2.0, 3.0];
            let factor = SO3BetweenFactor::new_axis_precision(17, [1200, 1201], measurement, &axis_prec);

            let info = factor.information();
            for i in 0..3 {
                assert_eq!(info[(i, i)], axis_prec[i]);
            }
        }
    }

    // Vision factor tests
    mod vision_tests {
        use super::*;
        use crate::factors::vision::*;

        #[test]
        fn test_camera_intrinsics_simple() {
            let intrinsics = CameraIntrinsics::new_simple(500.0, 500.0, 320.0, 240.0);
            
            assert_eq!(intrinsics.fx, 500.0);
            assert_eq!(intrinsics.fy, 500.0);
            assert_eq!(intrinsics.cx, 320.0);
            assert_eq!(intrinsics.cy, 240.0);
            assert_eq!(intrinsics.k1, 0.0);
            assert_eq!(intrinsics.k2, 0.0);
            assert_eq!(intrinsics.p1, 0.0);
            assert_eq!(intrinsics.p2, 0.0);
        }

        #[test]
        fn test_camera_intrinsics_with_distortion() {
            let intrinsics = CameraIntrinsics::new_with_distortion(
                500.0, 500.0, 320.0, 240.0,
                0.1, 0.01, 0.001, 0.0001
            );
            
            assert_eq!(intrinsics.k1, 0.1);
            assert_eq!(intrinsics.k2, 0.01);
            assert_eq!(intrinsics.p1, 0.001);
            assert_eq!(intrinsics.p2, 0.0001);
        }

        #[test]
        fn test_camera_projection() {
            let intrinsics = CameraIntrinsics::new_simple(500.0, 500.0, 320.0, 240.0);
            let point_3d = Vector3::new(1.0, 2.0, 5.0);
            
            let projected = intrinsics.project(&point_3d).unwrap();
            
            // Expected: [fx * X/Z + cx, fy * Y/Z + cy]
            let expected_u = 500.0 * 1.0 / 5.0 + 320.0; // = 100 + 320 = 420
            let expected_v = 500.0 * 2.0 / 5.0 + 240.0; // = 200 + 240 = 440
            
            assert!((projected.x - expected_u).abs() < 1e-10);
            assert!((projected.y - expected_v).abs() < 1e-10);
        }

        #[test]
        fn test_camera_projection_negative_z() {
            let intrinsics = CameraIntrinsics::new_simple(500.0, 500.0, 320.0, 240.0);
            let point_3d = Vector3::new(1.0, 2.0, -1.0); // Negative Z
            
            let result = intrinsics.project(&point_3d);
            assert!(result.is_err());
        }

        #[test]
        fn test_projection_factor_creation() {
            let intrinsics = CameraIntrinsics::new_simple(500.0, 500.0, 320.0, 240.0);
            let observation = Vector2::new(100.0, 200.0);
            let information = Matrix2::identity();
            
            let factor = ProjectionFactor::new(20, 1500, 2000, observation, intrinsics.clone(), information);
            
            assert_eq!(factor.id(), 20);
            assert_eq!(factor.observation(), &observation);
            assert_eq!(factor.intrinsics(), &intrinsics);
        }

        #[test]
        fn test_projection_factor_identity() {
            let intrinsics = CameraIntrinsics::new_simple(500.0, 500.0, 320.0, 240.0);
            let observation = Vector2::new(100.0, 200.0);
            
            let factor = ProjectionFactor::new_identity(21, 1600, 2100, observation, intrinsics);
            
            let expected_info = Matrix2::identity();
            assert_eq!(factor.information(), &expected_info);
        }

        #[test]
        fn test_projection_factor_isotropic() {
            let intrinsics = CameraIntrinsics::new_simple(500.0, 500.0, 320.0, 240.0);
            let observation = Vector2::new(100.0, 200.0);
            let precision = 4.0;
            
            let factor = ProjectionFactor::new_isotropic(22, 1700, 2200, observation, intrinsics, precision);
            
            let expected_info = Matrix2::identity() * precision;
            assert_eq!(factor.information(), &expected_info);
        }

        #[test]
        fn test_stereo_factor_creation() {
            let left_intrinsics = CameraIntrinsics::new_simple(500.0, 500.0, 320.0, 240.0);
            let right_intrinsics = CameraIntrinsics::new_simple(500.0, 500.0, 320.0, 240.0);
            let left_obs = Vector2::new(100.0, 200.0);
            let right_obs = Vector2::new(90.0, 200.0);
            let baseline = 0.12; // 12cm baseline
            let information = DMatrix::identity(4, 4);
            
            let factor = StereoFactor::new(
                23, 1800, 2300, left_obs, right_obs,
                left_intrinsics, right_intrinsics, baseline, information.clone()
            );
            
            assert_eq!(factor.id(), 23);
            assert_eq!(factor.information(), &information);
        }

        #[test]
        fn test_stereo_factor_identity() {
            let left_intrinsics = CameraIntrinsics::new_simple(500.0, 500.0, 320.0, 240.0);
            let right_intrinsics = CameraIntrinsics::new_simple(500.0, 500.0, 320.0, 240.0);
            let left_obs = Vector2::new(100.0, 200.0);
            let right_obs = Vector2::new(90.0, 200.0);
            let baseline = 0.12;
            
            let factor = StereoFactor::new_identity(
                24, 1900, 2400, left_obs, right_obs,
                left_intrinsics, right_intrinsics, baseline
            );
            
            let expected_info = DMatrix::identity(4, 4);
            assert_eq!(factor.information(), &expected_info);
        }
    }

    // Motion factor tests
    mod motion_tests {
        use super::*;
        use crate::factors::motion::*;

        #[test]
        fn test_odometry_factor_creation() {
            let measurement = create_test_se3();
            let information = DMatrix::identity(6, 6);
            let factor = OdometryFactor::new(30, [3000, 3001], measurement.clone(), information.clone());

            assert_eq!(factor.id(), 30);
            assert_eq!(factor.variable_keys(), &[3000, 3001]);
            assert_eq!(factor.measurement(), &measurement);
            assert_eq!(factor.information(), &information);
            assert_eq!(factor.dt(), None);
        }

        #[test]
        fn test_odometry_factor_with_time() {
            let measurement = create_test_se2();
            let information = DMatrix::identity(3, 3);
            let dt = 0.1;
            let factor = OdometryFactor::new_with_time(31, [3100, 3101], measurement, information, dt);

            assert_eq!(factor.dt(), Some(dt));
        }

        #[test]
        fn test_se2_odometry_factor_diagonal() {
            let measurement = create_test_se2();
            let trans_prec = 10.0;
            let rot_prec = 5.0;
            let factor = SE2OdometryFactor::new_diagonal(32, [3200, 3201], measurement, trans_prec, rot_prec);

            let info = factor.information();
            assert_eq!(info[(0, 0)], trans_prec);
            assert_eq!(info[(1, 1)], trans_prec);
            assert_eq!(info[(2, 2)], rot_prec);
        }

        #[test]
        fn test_se3_odometry_factor_diagonal() {
            let measurement = create_test_se3();
            let trans_prec = 10.0;
            let rot_prec = 5.0;
            let factor = SE3OdometryFactor::new_diagonal(33, [3300, 3301], measurement, trans_prec, rot_prec);

            let info = factor.information();
            for i in 0..3 {
                assert_eq!(info[(i, i)], trans_prec);
            }
            for i in 3..6 {
                assert_eq!(info[(i, i)], rot_prec);
            }
        }

        #[test]
        fn test_constant_velocity_factor_creation() {
            let information = DMatrix::identity(6, 6);
            let factor = ConstantVelocityFactor::<SE3>::new(40, [4000, 4001, 4002], information.clone());

            assert_eq!(factor.id(), 40);
            assert_eq!(factor.variable_keys(), &[4000, 4001, 4002]);
            assert_eq!(factor.information(), &information);
            assert_eq!(factor.time_intervals(), None);
        }

        #[test]
        fn test_constant_velocity_factor_with_time() {
            let information = DMatrix::identity(3, 3);
            let time_intervals = [0.1, 0.1];
            let factor = ConstantVelocityFactor::<SE2>::new_with_time(41, [4100, 4101, 4102], information, time_intervals);

            assert_eq!(factor.time_intervals(), Some(time_intervals));
        }

        #[test]
        fn test_velocity_factor_creation() {
            let velocity = SE2::identity().log(None); // SE2 tangent space
            let dt = 0.1;
            let information = DMatrix::identity(3, 3);
            let factor = VelocityFactor::<SE2>::new(50, [5000, 5001], velocity, dt, information.clone());

            assert_eq!(factor.id(), 50);
            assert_eq!(factor.variable_keys(), &[5000, 5001]);
            assert_eq!(factor.dt(), dt);
            assert_eq!(factor.information(), &information);
        }
    }

    // Robust kernel tests
    mod robust_tests {
        use crate::factors::robust::*;

        #[test]
        fn test_l2_kernel() {
            let kernel = L2Kernel::new();
            let x = 4.0;

            assert_eq!(kernel.rho(x), x);
            assert_eq!(kernel.psi(x), 1.0);
            assert_eq!(kernel.weight(x), 1.0 / x);
        }

        #[test]
        fn test_l2_kernel_zero_input() {
            let kernel = L2Kernel::new();
            let x = 0.0;

            assert_eq!(kernel.rho(x), 0.0);
            assert_eq!(kernel.psi(x), 1.0);
            assert_eq!(kernel.weight(x), 1.0); // Should handle division by zero
        }

        #[test]
        fn test_huber_kernel_small_error() {
            let delta = 1.345;
            let kernel = HuberKernel::new(delta);
            let x = 1.0; // x < delta²

            assert_eq!(kernel.rho(x), x);
            assert_eq!(kernel.psi(x), 1.0);
            assert_eq!(kernel.weight(x), 1.0 / x);
        }

        #[test]
        fn test_huber_kernel_large_error() {
            let delta = 1.345;
            let kernel = HuberKernel::new(delta);
            let x = 4.0; // x > delta²

            let expected_rho = 2.0 * delta * x.sqrt() - delta * delta;
            let expected_psi = delta / x.sqrt();
            let expected_weight = delta / (x * x.sqrt());

            assert!((kernel.rho(x) - expected_rho).abs() < 1e-10);
            assert!((kernel.psi(x) - expected_psi).abs() < 1e-10);
            assert!((kernel.weight(x) - expected_weight).abs() < 1e-10);
        }

        #[test]
        fn test_huber_kernel_default() {
            let kernel = HuberKernel::default();
            assert_eq!(kernel.parameter(), 1.345);
        }

        #[test]
        fn test_cauchy_kernel() {
            let sigma = 1.0;
            let kernel = CauchyKernel::new(sigma);
            let x = 2.0;

            let sigma_sq = sigma * sigma;
            let expected_rho = (sigma_sq / 2.0) * (1.0 + x / sigma_sq).ln();
            let expected_psi = 1.0 / (2.0 * (1.0 + x / sigma_sq));
            let expected_weight = 1.0 / (2.0 * x * (1.0 + x / sigma_sq));

            assert!((kernel.rho(x) - expected_rho).abs() < 1e-10);
            assert!((kernel.psi(x) - expected_psi).abs() < 1e-10);
            assert!((kernel.weight(x) - expected_weight).abs() < 1e-10);
        }

        #[test]
        fn test_tukey_kernel_small_error() {
            let c = 4.685;
            let kernel = TukeyKernel::new(c);
            let x = 2.0; // x < c²

            let c_sq = c * c;
            let ratio = x / c_sq;
            let term = 1.0 - ratio;
            let expected_rho = (c_sq / 6.0) * (1.0 - term * term * term);
            let expected_psi = term * term;
            let expected_weight = term * term / x;

            assert!((kernel.rho(x) - expected_rho).abs() < 1e-10);
            assert!((kernel.psi(x) - expected_psi).abs() < 1e-10);
            assert!((kernel.weight(x) - expected_weight).abs() < 1e-10);
        }

        #[test]
        fn test_tukey_kernel_large_error() {
            let c = 4.685;
            let kernel = TukeyKernel::new(c);
            let x = 30.0; // x > c²

            let expected_rho = c * c / 6.0;
            let expected_psi = 0.0;
            let expected_weight = 0.0;

            assert_eq!(kernel.rho(x), expected_rho);
            assert_eq!(kernel.psi(x), expected_psi);
            assert_eq!(kernel.weight(x), expected_weight);
        }

        #[test]
        fn test_geman_mcclure_kernel() {
            let sigma = 1.0;
            let kernel = GemanMcClureKernel::new(sigma);
            let x = 2.0;

            let sigma_sq = sigma * sigma;
            let expected_rho = x / (1.0 + x / sigma_sq);
            let denominator = 1.0 + x / sigma_sq;
            let expected_psi = 1.0 / (denominator * denominator);
            let expected_weight = 1.0 / (x * denominator * denominator);

            assert!((kernel.rho(x) - expected_rho).abs() < 1e-10);
            assert!((kernel.psi(x) - expected_psi).abs() < 1e-10);
            assert!((kernel.weight(x) - expected_weight).abs() < 1e-10);
        }

        #[test]
        fn test_kernel_parameter_setting() {
            let mut kernel = HuberKernel::new(1.0);
            assert_eq!(kernel.parameter(), 1.0);

            kernel.set_parameter(2.0);
            assert_eq!(kernel.parameter(), 2.0);
        }
    }
}
