//! Simple Pinhole Camera Projection Factors
//!
//! This module provides pinhole camera projection factors without distortion.
//! These are simpler alternatives to RadTan factors, useful for:
//! - Initial bundle adjustment implementations
//! - Cameras with minimal distortion
//! - Testing and verification
//!
//! # Camera Model
//!
//! The pinhole camera model projects 3D points to 2D image coordinates:
//! ```text
//! x_norm = X / Z
//! y_norm = Y / Z
//! u = fx * x_norm + cx
//! v = fy * y_norm + cy
//! ```
//!
//! # Reference
//!
//! Based on granite pinhole camera implementation:
//! https://github.com/DLR-RM/granite/blob/master/thirdparty/granite-headers/include/granite/camera/pinhole_camera.hpp

use super::Factor;
use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix2x4, Matrix3, Matrix3x6, Vector2, Vector3};

/// Compute pinhole projection residual
///
/// # Arguments
///
/// * `point_3d` - 3D point in camera frame [X, Y, Z]
/// * `point_2d_observed` - Observed 2D pixel coordinates [u, v]
/// * `camera_params` - Camera intrinsics [fx, fy, cx, cy]
/// * `use_bal_convention` - If true, use BAL/OpenGL convention (camera looks down -Z)
///
/// # Returns
///
/// `Some(residual)` if projection is valid, `None` otherwise
///
/// # Coordinate Conventions
///
/// - **Standard (use_bal_convention = false)**: Camera looks down +Z axis, Z > 0 is in front
/// - **BAL/OpenGL (use_bal_convention = true)**: Camera looks down -Z axis, Z < 0 is in front
#[inline]
fn compute_pinhole_residual(
    point_3d: &Vector3<f64>,
    point_2d_observed: &Vector2<f64>,
    camera_params: &DVector<f64>,
    use_bal_convention: bool,
) -> Option<Vector2<f64>> {
    let fx = camera_params[0];
    let fy = camera_params[1];
    let cx = camera_params[2];
    let cy = camera_params[3];

    let x = point_3d[0];
    let y = point_3d[1];
    let z = point_3d[2];

    // Check if point is behind camera based on convention
    let valid = if use_bal_convention {
        // BAL/OpenGL: camera looks down -Z, so valid points have Z < 0
        z < -f64::EPSILON
    } else {
        // Standard: camera looks down +Z, so valid points have Z > 0
        z > f64::EPSILON
    };

    if !valid {
        return None;
    }

    // Normalize coordinates
    // BAL uses: p = -P / P.z (the negative sign flips the coordinate system)
    let (x_norm, y_norm) = if use_bal_convention {
        (-x / z, -y / z)
    } else {
        (x / z, y / z)
    };

    // Project to image plane
    let u = fx * x_norm + cx;
    let v = fy * y_norm + cy;

    // Compute residual (predicted - observed)
    Some(Vector2::new(
        u - point_2d_observed[0],
        v - point_2d_observed[1],
    ))
}

/// Compute Jacobian of projection w.r.t. camera intrinsics [fx, fy, cx, cy]
///
/// # Derivation
///
/// ```text
/// u = fx * (X/Z) + cx
/// v = fy * (Y/Z) + cy
///
/// ∂u/∂fx = X/Z = x_norm
/// ∂u/∂fy = 0
/// ∂u/∂cx = 1
/// ∂u/∂cy = 0
///
/// ∂v/∂fx = 0
/// ∂v/∂fy = Y/Z = y_norm
/// ∂v/∂cx = 0
/// ∂v/∂cy = 1
/// ```
#[inline]
fn compute_camera_params_jacobian(point_3d: &Vector3<f64>) -> Matrix2x4<f64> {
    let x = point_3d[0];
    let y = point_3d[1];
    let z = point_3d[2];

    let x_norm = x / z;
    let y_norm = y / z;

    Matrix2x4::new(
        x_norm, 0.0, 1.0, 0.0, // ∂u/∂[fx, fy, cx, cy]
        0.0, y_norm, 0.0, 1.0, // ∂v/∂[fx, fy, cx, cy]
    )
}

/// Compute Jacobian of projection w.r.t. 3D point [X, Y, Z]
///
/// # Derivation
///
/// Standard convention (camera looks down +Z):
/// ```text
/// u = fx * (X/Z) + cx
/// v = fy * (Y/Z) + cy
///
/// ∂u/∂X = fx / Z
/// ∂u/∂Y = 0
/// ∂u/∂Z = -fx * X / Z²
///
/// ∂v/∂X = 0
/// ∂v/∂Y = fy / Z
/// ∂v/∂Z = -fy * Y / Z²
/// ```
///
/// BAL convention (camera looks down -Z, using p = -P/P.z):
/// ```text
/// u = fx * (-X/Z) + cx
/// v = fy * (-Y/Z) + cy
///
/// ∂u/∂X = -fx / Z
/// ∂u/∂Y = 0
/// ∂u/∂Z = fx * X / Z²
///
/// ∂v/∂X = 0
/// ∂v/∂Y = -fy / Z
/// ∂v/∂Z = fy * Y / Z²
/// ```
#[inline]
fn compute_point_jacobian(
    point_3d: &Vector3<f64>,
    camera_params: &DVector<f64>,
    use_bal_convention: bool,
) -> Matrix2x3<f64> {
    let fx = camera_params[0];
    let fy = camera_params[1];

    let x = point_3d[0];
    let y = point_3d[1];
    let z = point_3d[2];

    let z_inv = 1.0 / z;
    let z_inv_sq = z_inv * z_inv;

    if use_bal_convention {
        // BAL: p = -P / P.z
        Matrix2x3::new(
            -fx * z_inv,
            0.0,
            fx * x * z_inv_sq,
            0.0,
            -fy * z_inv,
            fy * y * z_inv_sq,
        )
    } else {
        // Standard: p = P / P.z
        Matrix2x3::new(
            fx * z_inv,
            0.0,
            -fx * x * z_inv_sq,
            0.0,
            fy * z_inv,
            -fy * y * z_inv_sq,
        )
    }
}

/// Convert axis-angle vector to rotation matrix using SO3 infrastructure
///
/// # Arguments
///
/// * `axis_angle` - Rotation vector [rx, ry, rz] where magnitude is angle
///
/// # Returns
///
/// 3×3 rotation matrix
#[inline]
fn axis_angle_to_rotation_matrix(axis_angle: &Vector3<f64>) -> Matrix3<f64> {
    use crate::manifold::so3::SO3;
    SO3::from_scaled_axis(*axis_angle).rotation_matrix()
}

/// Transform 3D point from world to camera coordinates
///
/// Applies the camera transformation: P_cam = R(axis_angle) * P_world + t
///
/// # Arguments
///
/// * `point_world` - 3D point in world frame [X_w, Y_w, Z_w]
/// * `axis_angle` - Camera rotation as axis-angle [rx, ry, rz]
/// * `translation` - Camera translation [tx, ty, tz]
///
/// # Returns
///
/// 3D point in camera frame [X_c, Y_c, Z_c]
#[inline]
fn transform_point_to_camera(
    point_world: &Vector3<f64>,
    axis_angle: &Vector3<f64>,
    translation: &Vector3<f64>,
) -> Vector3<f64> {
    let R = axis_angle_to_rotation_matrix(axis_angle);
    R * point_world + translation
}

/// Compute Jacobian of camera transformation w.r.t. camera pose
///
/// Returns 3×6 Jacobian: [∂P_cam/∂[rx,ry,rz,tx,ty,tz]]
///
/// For the transformation P_cam = R(axis_angle) * P_world + t:
/// - ∂P_cam/∂t = I (identity, last 3 columns)
/// - ∂P_cam/∂r = ∂(R*p)/∂r (rotation derivatives, first 3 columns)
///
/// The rotation part is computed using finite differences for simplicity.
/// This can be optimized later with analytical SO3 adjoint formulas.
#[inline]
fn compute_pose_jacobian(
    point_world: &Vector3<f64>,
    axis_angle: &Vector3<f64>,
    translation: &Vector3<f64>,
) -> Matrix3x6<f64> {
    // Translation part: ∂P_cam/∂t = I (last 3 columns)
    let translation_jac = Matrix3::<f64>::identity();

    // Rotation part: ∂(R*p)/∂r using finite differences (first 3 columns)
    let epsilon = 1e-8;
    let mut rotation_jac = Matrix3::<f64>::zeros();

    for i in 0..3 {
        let mut axis_angle_plus = *axis_angle;
        let mut axis_angle_minus = *axis_angle;
        axis_angle_plus[i] += epsilon;
        axis_angle_minus[i] -= epsilon;

        let p_plus = transform_point_to_camera(point_world, &axis_angle_plus, translation);
        let p_minus = transform_point_to_camera(point_world, &axis_angle_minus, translation);

        let derivative = (p_plus - p_minus) / (2.0 * epsilon);
        rotation_jac.set_column(i, &derivative);
    }

    // Combine: [∂P/∂r | ∂P/∂t]
    Matrix3x6::from_columns(&[
        rotation_jac.column(0).into_owned(),
        rotation_jac.column(1).into_owned(),
        rotation_jac.column(2).into_owned(),
        translation_jac.column(0).into_owned(),
        translation_jac.column(1).into_owned(),
        translation_jac.column(2).into_owned(),
    ])
}

/// Factor for optimizing camera intrinsic parameters with fixed 3D points
///
/// This factor holds 3D points constant and optimizes camera parameters [fx, fy, cx, cy].
/// Useful for camera calibration where point positions are known.
#[derive(Debug, Clone)]
pub struct PinholeCameraParamsFactor {
    /// 3D points in camera coordinate system (3×N matrix)
    pub points_3d: Vec<Vector3<f64>>,
    /// Corresponding observed 2D points in image coordinates (2×N)
    pub points_2d: Vec<Vector2<f64>>,
}

impl PinholeCameraParamsFactor {
    /// Create a new camera parameters factor
    ///
    /// # Arguments
    ///
    /// * `points_3d` - Vector of 3D points in camera frame
    /// * `points_2d` - Vector of corresponding 2D observations
    pub fn new(points_3d: Vec<Vector3<f64>>, points_2d: Vec<Vector2<f64>>) -> Self {
        assert_eq!(
            points_3d.len(),
            points_2d.len(),
            "Must have same number of 3D and 2D points"
        );
        Self {
            points_3d,
            points_2d,
        }
    }

    /// Get the number of point correspondences
    pub fn num_points(&self) -> usize {
        self.points_3d.len()
    }
}

impl Factor for PinholeCameraParamsFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        assert_eq!(
            params.len(),
            1,
            "Expected 1 parameter block (camera params)"
        );
        let camera_params = &params[0];
        assert_eq!(
            camera_params.len(),
            4,
            "Camera params must be [fx, fy, cx, cy]"
        );

        let num_points = self.num_points();
        let residual_dim = 2 * num_points;

        let mut residuals = DVector::zeros(residual_dim);
        let mut jacobian = if compute_jacobian {
            Some(DMatrix::zeros(residual_dim, 4))
        } else {
            None
        };

        for i in 0..num_points {
            if let Some(res) = compute_pinhole_residual(
                &self.points_3d[i],
                &self.points_2d[i],
                camera_params,
                false,
            ) {
                residuals[2 * i] = res[0];
                residuals[2 * i + 1] = res[1];

                if let Some(ref mut jac) = jacobian {
                    let jac_block = compute_camera_params_jacobian(&self.points_3d[i]);
                    jac.fixed_view_mut::<2, 4>(2 * i, 0).copy_from(&jac_block);
                }
            } else {
                // Point behind camera - set large residual
                residuals[2 * i] = 1e6;
                residuals[2 * i + 1] = 1e6;
            }
        }

        (residuals, jacobian)
    }

    fn get_dimension(&self) -> usize {
        2 * self.num_points()
    }
}

/// Factor for optimizing 3D point positions with fixed camera parameters
///
/// This factor holds camera parameters constant and optimizes the 3D point position.
/// This is the standard factor for bundle adjustment where camera intrinsics are known.
#[derive(Debug, Clone)]
pub struct PinholeProjectionFactor {
    /// Observed 2D point in image coordinates [u, v]
    pub point_2d: Vector2<f64>,
    /// Fixed camera parameters [fx, fy, cx, cy]
    pub camera_params: DVector<f64>,
}

impl PinholeProjectionFactor {
    /// Create a new projection factor
    ///
    /// # Arguments
    ///
    /// * `point_2d` - Observed 2D pixel coordinates
    /// * `camera_params` - Camera intrinsics [fx, fy, cx, cy]
    pub fn new(point_2d: Vector2<f64>, camera_params: DVector<f64>) -> Self {
        assert_eq!(
            camera_params.len(),
            4,
            "Camera params must be [fx, fy, cx, cy]"
        );
        Self {
            point_2d,
            camera_params,
        }
    }
}

impl Factor for PinholeProjectionFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        assert_eq!(params.len(), 1, "Expected 1 parameter block (3D point)");
        let point_3d_vec = &params[0];
        assert_eq!(point_3d_vec.len(), 3, "3D point must have 3 coordinates");

        let point_3d = Vector3::new(point_3d_vec[0], point_3d_vec[1], point_3d_vec[2]);

        let residual = if let Some(res) =
            compute_pinhole_residual(&point_3d, &self.point_2d, &self.camera_params, false)
        {
            DVector::from_vec(vec![res[0], res[1]])
        } else {
            // Point behind camera
            DVector::from_vec(vec![1e6, 1e6])
        };

        let jacobian = if compute_jacobian {
            if point_3d[2] > f64::EPSILON {
                let jac_block = compute_point_jacobian(&point_3d, &self.camera_params, false);
                Some(DMatrix::from_iterator(2, 3, jac_block.iter().cloned()))
            } else {
                // Point behind camera - zero jacobian
                Some(DMatrix::zeros(2, 3))
            }
        } else {
            None
        };

        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        2
    }
}

/// Bundle Adjustment Factor connecting camera pose and 3D world point
///
/// This factor optimizes both camera pose and 3D landmark positions simultaneously.
/// It handles the full bundle adjustment pipeline:
/// 1. Transform world point to camera frame: P_cam = R(axis_angle) * P_world + t
/// 2. Project to image plane: [u, v] = pinhole_projection(P_cam)
/// 3. Compute residual: r = [u, v] - observation
///
/// # Parameter Blocks
///
/// - Block 0: Camera pose [rx, ry, rz, tx, ty, tz] (6 DOF, axis-angle + translation)
/// - Block 1: 3D world point [X_w, Y_w, Z_w] (3 DOF)
///
/// # Residual Dimension
///
/// - 2D (u, v pixel error)
#[derive(Debug, Clone)]
pub struct BundleAdjustmentFactor {
    /// Observed 2D point in image coordinates [u, v]
    pub observation_2d: Vector2<f64>,
    /// Fixed camera intrinsics [fx, fy, cx, cy]
    pub camera_intrinsics: DVector<f64>,
}

impl BundleAdjustmentFactor {
    /// Create a new bundle adjustment factor
    ///
    /// # Arguments
    ///
    /// * `observation_2d` - Observed pixel coordinates [u, v]
    /// * `camera_intrinsics` - Camera parameters [fx, fy, cx, cy]
    pub fn new(observation_2d: Vector2<f64>, camera_intrinsics: DVector<f64>) -> Self {
        assert_eq!(
            camera_intrinsics.len(),
            4,
            "Camera intrinsics must be [fx, fy, cx, cy]"
        );
        Self {
            observation_2d,
            camera_intrinsics,
        }
    }
}

impl Factor for BundleAdjustmentFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        assert_eq!(
            params.len(),
            2,
            "Expected 2 parameter blocks: [camera_pose(6), world_point(3)]"
        );

        let camera_pose = &params[0];
        let world_point = &params[1];

        assert_eq!(
            camera_pose.len(),
            6,
            "Camera pose must be 6 DOF [rx,ry,rz,tx,ty,tz]"
        );
        assert_eq!(world_point.len(), 3, "World point must be 3D [X,Y,Z]");

        // Extract pose components
        let axis_angle = Vector3::new(camera_pose[0], camera_pose[1], camera_pose[2]);
        let translation = Vector3::new(camera_pose[3], camera_pose[4], camera_pose[5]);
        let point_world = Vector3::new(world_point[0], world_point[1], world_point[2]);

        // Step 1: Transform world point to camera frame
        let point_camera = transform_point_to_camera(&point_world, &axis_angle, &translation);

        // Step 2: Project to image plane and compute residual
        // Use BAL convention (camera looks down -Z axis)
        let residual = if let Some(res) = compute_pinhole_residual(
            &point_camera,
            &self.observation_2d,
            &self.camera_intrinsics,
            true,
        ) {
            DVector::from_vec(vec![res[0], res[1]])
        } else {
            // Point behind camera - large residual
            DVector::from_vec(vec![1e6, 1e6])
        };

        // Step 3: Compute Jacobians if requested
        let jacobian = if compute_jacobian && point_camera[2] < -f64::EPSILON {
            // Chain rule: ∂r/∂pose = (∂r/∂P_cam) × (∂P_cam/∂pose)
            //             ∂r/∂point = (∂r/∂P_cam) × (∂P_cam/∂point)

            // ∂r/∂P_cam: 2×3 Jacobian (projection w.r.t. camera point, BAL convention)
            let jac_projection =
                compute_point_jacobian(&point_camera, &self.camera_intrinsics, true);

            // ∂P_cam/∂pose: 3×6 Jacobian (camera transform w.r.t. pose)
            let jac_pose_transform = compute_pose_jacobian(&point_world, &axis_angle, &translation);

            // ∂P_cam/∂point: 3×3 Jacobian (just rotation matrix R)
            let r = axis_angle_to_rotation_matrix(&axis_angle);

            // Chain rule multiplication
            let jac_pose = jac_projection * jac_pose_transform; // 2×6
            let jac_point = jac_projection * r; // 2×3

            // Concatenate: [2×6 | 2×3] = 2×9 total Jacobian
            let mut full_jacobian = DMatrix::zeros(2, 9);
            full_jacobian.view_mut((0, 0), (2, 6)).copy_from(&jac_pose);
            full_jacobian.view_mut((0, 6), (2, 3)).copy_from(&jac_point);

            Some(full_jacobian)
        } else {
            // Point behind camera - zero Jacobian
            Some(DMatrix::zeros(2, 9))
        };

        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        2 // 2D pixel error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinhole_projection_simple() {
        // Simple test: point at [0, 0, 1] with identity-like camera
        let point_3d = Vector3::new(0.0, 0.0, 1.0);
        let camera_params = DVector::from_vec(vec![1.0, 1.0, 0.0, 0.0]); // fx=1, fy=1, cx=0, cy=0
        let observed = Vector2::new(0.0, 0.0);

        let residual = compute_pinhole_residual(&point_3d, &observed, &camera_params, false);
        assert!(residual.is_some());
        let res = residual.unwrap();
        assert!(res[0].abs() < 1e-10);
        assert!(res[1].abs() < 1e-10);
    }

    #[test]
    fn test_pinhole_projection_offset() {
        // Point at [1, 2, 2] should project to [0.5, 1.0] in normalized coords
        let point_3d = Vector3::new(1.0, 2.0, 2.0);
        let camera_params = DVector::from_vec(vec![100.0, 100.0, 50.0, 60.0]);
        // Expected: u = 100 * 0.5 + 50 = 100, v = 100 * 1.0 + 60 = 160
        let observed = Vector2::new(100.0, 160.0);

        let residual = compute_pinhole_residual(&point_3d, &observed, &camera_params, false);
        assert!(residual.is_some());
        let res = residual.unwrap();
        assert!(res[0].abs() < 1e-10);
        assert!(res[1].abs() < 1e-10);
    }

    #[test]
    fn test_point_behind_camera() {
        let point_3d = Vector3::new(1.0, 2.0, -1.0); // Negative Z
        let camera_params = DVector::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        let observed = Vector2::new(0.0, 0.0);

        let residual = compute_pinhole_residual(&point_3d, &observed, &camera_params, false);
        assert!(residual.is_none());
    }

    #[test]
    fn test_projection_factor_linearize() {
        let point_2d = Vector2::new(100.0, 160.0);
        let camera_params = DVector::from_vec(vec![100.0, 100.0, 50.0, 60.0]);
        let factor = PinholeProjectionFactor::new(point_2d, camera_params);

        let point_3d = DVector::from_vec(vec![1.0, 2.0, 2.0]);
        let (residual, jacobian) = factor.linearize(&[point_3d], true);

        assert_eq!(residual.len(), 2);
        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();
        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 3);

        // Residual should be near zero for this perfect projection
        assert!(residual[0].abs() < 1e-10);
        assert!(residual[1].abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_finite_difference() {
        // Verify analytical Jacobian matches finite differences
        let point_3d = Vector3::new(1.5, 2.5, 3.0);
        let camera_params = DVector::from_vec(vec![100.0, 100.0, 50.0, 60.0]);
        let observed = Vector2::new(100.0, 150.0);

        let analytical_jac = compute_point_jacobian(&point_3d, &camera_params, false);

        // Finite difference step
        let epsilon = 1e-6;

        for i in 0..3 {
            let mut point_plus = point_3d;
            let mut point_minus = point_3d;
            point_plus[i] += epsilon;
            point_minus[i] -= epsilon;

            let res_plus =
                compute_pinhole_residual(&point_plus, &observed, &camera_params, false).unwrap();
            let res_minus =
                compute_pinhole_residual(&point_minus, &observed, &camera_params, false).unwrap();

            let fd_jac_col = (res_plus - res_minus) / (2.0 * epsilon);

            assert!((analytical_jac[(0, i)] - fd_jac_col[0]).abs() < 1e-5);
            assert!((analytical_jac[(1, i)] - fd_jac_col[1]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_camera_params_factor() {
        let points_3d = vec![Vector3::new(1.0, 2.0, 2.0), Vector3::new(-1.0, 1.0, 3.0)];
        let points_2d = vec![Vector2::new(100.0, 160.0), Vector2::new(16.67, 93.33)];
        let factor = PinholeCameraParamsFactor::new(points_3d, points_2d);

        let camera_params = DVector::from_vec(vec![100.0, 100.0, 50.0, 60.0]);
        let (residual, jacobian) = factor.linearize(&[camera_params], true);

        assert_eq!(residual.len(), 4); // 2 points × 2 dimensions
        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();
        assert_eq!(jac.nrows(), 4);
        assert_eq!(jac.ncols(), 4);
    }

    #[test]
    fn test_axis_angle_to_rotation() {
        // Test identity rotation (zero axis-angle)
        let axis_angle = Vector3::zeros();
        let R = axis_angle_to_rotation_matrix(&axis_angle);
        assert!((R - Matrix3::identity()).norm() < 1e-10);

        // Test 90° rotation around Z-axis
        let axis_angle = Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        let R = axis_angle_to_rotation_matrix(&axis_angle);
        let point = Vector3::new(1.0, 0.0, 0.0);
        let rotated = R * point;

        // Should rotate X-axis to Y-axis
        assert!((rotated - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_transform_point_to_camera() {
        // Test identity transformation
        let point_world = Vector3::new(1.0, 2.0, 3.0);
        let axis_angle = Vector3::zeros();
        let translation = Vector3::zeros();
        let point_cam = transform_point_to_camera(&point_world, &axis_angle, &translation);
        assert!((point_cam - point_world).norm() < 1e-10);

        // Test pure translation
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let point_cam = transform_point_to_camera(&point_world, &axis_angle, &translation);
        assert!((point_cam - (point_world + translation)).norm() < 1e-10);
    }

    #[test]
    fn test_bundle_adjustment_factor_residual() {
        // Simple test: identity camera, point at [0, 0, -1] (BAL convention: negative Z in front)
        let observation = Vector2::new(0.0, 0.0);
        let camera_intrinsics = DVector::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        let factor = BundleAdjustmentFactor::new(observation, camera_intrinsics);

        // Camera pose: identity (no rotation, no translation)
        let camera_pose = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // World point at [0, 0, -1] (BAL convention)
        let world_point = DVector::from_vec(vec![0.0, 0.0, -1.0]);

        let (residual, jacobian) = factor.linearize(&[camera_pose, world_point], true);

        assert_eq!(residual.len(), 2);
        assert!(residual[0].abs() < 1e-10, "residual[0] = {}", residual[0]);
        assert!(residual[1].abs() < 1e-10, "residual[1] = {}", residual[1]);

        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();
        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 9); // 6 (pose) + 3 (point)
    }

    #[test]
    fn test_bundle_adjustment_jacobian_finite_diff() {
        // Verify Jacobians with finite differences
        let observation = Vector2::new(50.0, 60.0);
        let camera_intrinsics = DVector::from_vec(vec![100.0, 100.0, 0.0, 0.0]);
        let factor = BundleAdjustmentFactor::new(observation, camera_intrinsics);

        let camera_pose = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.5, 0.6, 0.7]);
        let world_point = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let (_, jacobian_opt) = factor.linearize(&[camera_pose.clone(), world_point.clone()], true);
        let jac = jacobian_opt.unwrap();

        let epsilon = 1e-6;

        // Check pose Jacobian (first 6 columns)
        for i in 0..6 {
            let mut pose_plus = camera_pose.clone();
            let mut pose_minus = camera_pose.clone();
            pose_plus[i] += epsilon;
            pose_minus[i] -= epsilon;

            let (res_plus, _) = factor.linearize(&[pose_plus, world_point.clone()], false);
            let (res_minus, _) = factor.linearize(&[pose_minus, world_point.clone()], false);

            let fd_col = (res_plus - res_minus) / (2.0 * epsilon);

            assert!(
                (jac[(0, i)] - fd_col[0]).abs() < 1e-4,
                "Pose Jacobian mismatch at column {}",
                i
            );
            assert!(
                (jac[(1, i)] - fd_col[1]).abs() < 1e-4,
                "Pose Jacobian mismatch at column {}",
                i
            );
        }

        // Check point Jacobian (last 3 columns)
        for i in 0..3 {
            let mut point_plus = world_point.clone();
            let mut point_minus = world_point.clone();
            point_plus[i] += epsilon;
            point_minus[i] -= epsilon;

            let (res_plus, _) = factor.linearize(&[camera_pose.clone(), point_plus], false);
            let (res_minus, _) = factor.linearize(&[camera_pose.clone(), point_minus], false);

            let fd_col = (res_plus - res_minus) / (2.0 * epsilon);

            assert!(
                (jac[(0, 6 + i)] - fd_col[0]).abs() < 1e-4,
                "Point Jacobian mismatch at column {}",
                i
            );
            assert!(
                (jac[(1, 6 + i)] - fd_col[1]).abs() < 1e-4,
                "Point Jacobian mismatch at column {}",
                i
            );
        }
    }
}
