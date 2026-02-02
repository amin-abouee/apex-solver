//! Shared utilities for camera integration tests
//!
//! This module provides common functionality for testing camera projection factors
//! with self-calibration (optimizing pose, landmarks, and intrinsics simultaneously).

use apex_manifolds::se3::{SE3Tangent, SE3};
use apex_manifolds::so3::SO3;
use apex_manifolds::LieGroup;
use nalgebra::{DVector, Vector3};

/// Generate N 3D points in a realistic scene (hemisphere in front of camera)
///
/// Points are distributed in a hemisphere with:
/// - Depth: 2-5 meters (realistic for SLAM/SfM scenarios)
/// - X/Y spread: ±2 meters
/// - All points in front of camera (positive Z)
///
/// # Arguments
/// * `n` - Number of points to generate
/// * `_seed` - Random seed (unused, for API compatibility)
///
/// # Returns
/// Vector of 3D points in camera frame
pub fn generate_scene_points(n: usize, _seed: u64) -> Vec<Vector3<f64>> {
    (0..n)
        .map(|i| {
            // Deterministic but well-distributed points based on index
            let angle1 = (i as f64 * 2.4) % (2.0 * std::f64::consts::PI);
            let angle2 = (i as f64 * 1.7) % std::f64::consts::PI;

            // Depth between 2-5 meters
            let depth = 2.0 + 3.0 * ((i as f64 * 0.17) % 1.0);

            // Convert to Cartesian (hemisphere in front of camera)
            let x = depth * angle2.sin() * angle1.cos();
            let y = depth * angle2.sin() * angle1.sin();
            let z = depth * angle2.cos().abs() + 2.0; // Ensure z > 0 (in front)

            Vector3::new(x, y, z)
        })
        .collect()
}

/// Generate Gaussian-like noise using Box-Muller transform
///
/// # Arguments
/// * `mean` - Mean of distribution
/// * `std_dev` - Standard deviation
/// * `index` - Deterministic index for reproducibility
///
/// # Returns
/// Pseudo-random value from normal distribution
fn generate_normal(mean: f64, std_dev: f64, index: usize) -> f64 {
    // Box-Muller transform for Gaussian noise
    let u1 = ((index * 12345 + 67890) % 10000) as f64 / 10000.0;
    let u2 = ((index * 54321 + 98765) % 10000) as f64 / 10000.0;

    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + std_dev * z0
}

/// Add Gaussian noise to SE3 pose
///
/// Noise is added in tangent space (6-DOF):
/// - Translation: 3 DOF with specified standard deviation
/// - Rotation: 3 DOF (axis-angle) with specified standard deviation in degrees
///
/// # Arguments
/// * `pose` - Ground truth pose
/// * `translation_std` - Standard deviation for translation noise (meters)
/// * `rotation_std_deg` - Standard deviation for rotation noise (degrees)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Perturbed pose
pub fn perturb_pose(pose: &SE3, translation_std: f64, rotation_std_deg: f64, seed: u64) -> SE3 {
    let base = seed as usize;

    // Generate translation noise (meters)
    let noise_translation = Vector3::new(
        generate_normal(0.0, translation_std, base),
        generate_normal(0.0, translation_std, base + 1),
        generate_normal(0.0, translation_std, base + 2),
    );

    // Generate rotation noise (radians)
    let rotation_std_rad = rotation_std_deg.to_radians();
    let noise_rotation = Vector3::new(
        generate_normal(0.0, rotation_std_rad, base + 3),
        generate_normal(0.0, rotation_std_rad, base + 4),
        generate_normal(0.0, rotation_std_rad, base + 5),
    );

    // Create tangent vector (order: translation, rotation)
    let delta = SE3Tangent::new(noise_translation, noise_rotation);

    // Apply perturbation using manifold plus operation
    pose.plus(&delta, None, None)
}

/// Add Gaussian noise to landmarks
///
/// Each landmark gets independent Gaussian noise in X, Y, Z directions.
///
/// # Arguments
/// * `landmarks` - Ground truth landmarks
/// * `std_dev` - Standard deviation for noise (meters)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Perturbed landmarks
pub fn perturb_landmarks(landmarks: &[Vector3<f64>], std_dev: f64, seed: u64) -> Vec<Vector3<f64>> {
    landmarks
        .iter()
        .enumerate()
        .map(|(i, pt)| {
            let base = (seed as usize) + i * 3;
            Vector3::new(
                pt.x + generate_normal(0.0, std_dev, base),
                pt.y + generate_normal(0.0, std_dev, base + 1),
                pt.z + generate_normal(0.0, std_dev, base + 2),
            )
        })
        .collect()
}

/// Add percentage-based noise to camera intrinsics
///
/// Each parameter is perturbed as: param_noisy = param_true * (1 + N(0, percent))
///
/// This models calibration uncertainty where relative errors are more meaningful
/// than absolute errors (e.g., 5% error in focal length vs. 5 pixel error).
///
/// # Arguments
/// * `intrinsics` - Ground truth intrinsic parameters
/// * `percent` - Noise level as fraction (e.g., 0.05 = 5%)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Perturbed intrinsics vector
pub fn perturb_intrinsics(intrinsics: &[f64], percent: f64, seed: u64) -> Vec<f64> {
    intrinsics
        .iter()
        .enumerate()
        .map(|(i, &param)| {
            let noise = generate_normal(0.0, percent, seed as usize + i);
            param * (1.0 + noise)
        })
        .collect()
}

/// Create flattened landmark vector for Problem
///
/// Converts Vec<Vector3<f64>> to DVector<f64> in format [x0, y0, z0, x1, y1, z1, ...]
/// This is the format expected by apex_solver's Problem for RN manifold variables.
///
/// # Arguments
/// * `landmarks` - Vector of 3D points
///
/// # Returns
/// Flattened DVector suitable for Problem initialization
pub fn flatten_landmarks(landmarks: &[Vector3<f64>]) -> DVector<f64> {
    let mut flat = Vec::with_capacity(landmarks.len() * 3);
    for pt in landmarks {
        flat.push(pt.x);
        flat.push(pt.y);
        flat.push(pt.z);
    }
    DVector::from_vec(flat)
}

/// Unflatten landmark vector from DVector to Vec<Vector3>
///
/// Inverse operation of flatten_landmarks - converts DVector [x0, y0, z0, ...]
/// back to Vec<Vector3<f64>>.
///
/// # Arguments
/// * `flat` - Flattened landmark vector
///
/// # Returns
/// Vector of 3D points
pub fn unflatten_landmarks(flat: &DVector<f64>) -> Vec<Vector3<f64>> {
    assert_eq!(
        flat.len() % 3,
        0,
        "Landmark vector length must be multiple of 3"
    );

    flat.as_slice()
        .chunks_exact(3)
        .map(|chunk| Vector3::new(chunk[0], chunk[1], chunk[2]))
        .collect()
}

/// Generate planar calibration target points on a wall at fixed Z depth.
///
/// Creates a regular grid of 3D points suitable for simulating calibration
/// patterns like ArUco markers or chessboard corners.
///
/// # Arguments
/// * `num_x` - Number of points along X axis
/// * `num_y` - Number of points along Y axis
/// * `spacing` - Distance between adjacent points (meters)
/// * `wall_z` - Z coordinate of the wall (distance from camera)
///
/// # Returns
/// Vector of 3D points arranged in a grid, centered at (0, 0, wall_z)
///
/// # Example
/// ```ignore
/// // 20x10 grid with 10cm spacing at 3m distance
/// let points = generate_wall_calibration_points(20, 10, 0.1, 3.0);
/// assert_eq!(points.len(), 200);
/// ```
pub fn generate_wall_calibration_points(
    num_x: usize,
    num_y: usize,
    spacing: f64,
    wall_z: f64,
) -> Vec<Vector3<f64>> {
    let mut points = Vec::with_capacity(num_x * num_y);

    // Center the grid around origin
    let offset_x = (num_x - 1) as f64 * spacing / 2.0;
    let offset_y = (num_y - 1) as f64 * spacing / 2.0;

    for iy in 0..num_y {
        for ix in 0..num_x {
            let x = ix as f64 * spacing - offset_x;
            let y = iy as f64 * spacing - offset_y;
            points.push(Vector3::new(x, y, wall_z));
        }
    }

    points
}

/// Generate camera poses distributed on a horizontal arc facing a target.
///
/// Creates camera poses suitable for multi-view calibration scenarios.
/// All cameras face the +Z direction (towards the wall/target).
///
/// # Arguments
/// * `num_cameras` - Number of cameras to generate
/// * `arc_spread` - Horizontal spread of the arc (half-width in meters)
/// * `_distance` - Reserved for future use (e.g., curved arc depth)
///
/// # Returns
/// Vector of SE3 poses, each representing a camera position and orientation
///
/// # Example
/// ```ignore
/// // 5 cameras spread over 0.6m (±0.3m from center)
/// let poses = generate_arc_camera_poses(5, 0.3, 3.0);
/// assert_eq!(poses.len(), 5);
/// ```
pub fn generate_arc_camera_poses(num_cameras: usize, arc_spread: f64, _distance: f64) -> Vec<SE3> {
    let mut poses = Vec::with_capacity(num_cameras);

    for i in 0..num_cameras {
        // Distribute cameras evenly across the arc
        // t ranges from -0.5 to +0.5
        let t = if num_cameras > 1 {
            (i as f64 / (num_cameras - 1) as f64) - 0.5
        } else {
            0.0
        };

        // Camera position on horizontal arc
        let x = t * 2.0 * arc_spread;
        let y = 0.0;
        let z = 0.0;

        // All cameras face +Z direction (identity rotation)
        // For a more realistic setup, we could add small rotations
        // to point towards the center of the wall
        let translation = Vector3::new(x, y, z);
        let rotation = SO3::identity();

        let pose = SE3::from_translation_so3(translation, rotation);
        poses.push(pose);
    }

    poses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_wall_calibration_points() {
        let points = generate_wall_calibration_points(20, 10, 0.1, 3.0);
        assert_eq!(points.len(), 200);

        // All points should be at z = 3.0
        for pt in &points {
            assert!(
                (pt.z - 3.0).abs() < 1e-10,
                "Point should be at z=3.0: {:?}",
                pt
            );
        }

        // Check centering: first and last points should be symmetric
        let first = &points[0];
        let last = &points[199];

        // First point: bottom-left corner
        assert!((first.x - (-0.95)).abs() < 1e-10); // (20-1)*0.1/2 = 0.95
        assert!((first.y - (-0.45)).abs() < 1e-10); // (10-1)*0.1/2 = 0.45

        // Last point: top-right corner
        assert!((last.x - 0.95).abs() < 1e-10);
        assert!((last.y - 0.45).abs() < 1e-10);
    }

    #[test]
    fn test_generate_arc_camera_poses() {
        let poses = generate_arc_camera_poses(5, 0.3, 3.0);
        assert_eq!(poses.len(), 5);

        // Check camera positions
        let translations: Vec<_> = poses.iter().map(|p| p.translation()).collect();

        // Camera 0 should be at x = -0.3 (leftmost)
        assert!((translations[0].x - (-0.3)).abs() < 1e-10);
        // Camera 2 should be at x = 0 (center)
        assert!(translations[2].x.abs() < 1e-10);
        // Camera 4 should be at x = 0.3 (rightmost)
        assert!((translations[4].x - 0.3).abs() < 1e-10);

        // All cameras at y = 0, z = 0
        for t in &translations {
            assert!(t.y.abs() < 1e-10);
            assert!(t.z.abs() < 1e-10);
        }
    }

    #[test]
    fn test_generate_scene_points() {
        let points = generate_scene_points(50, 42);
        assert_eq!(points.len(), 50);

        // All points should be in front of camera (z > 0)
        for pt in &points {
            assert!(pt.z > 0.0, "Point should be in front of camera: {:?}", pt);
            assert!(
                pt.z >= 2.0 && pt.z <= 7.0,
                "Depth should be reasonable: {}",
                pt.z
            );
        }
    }

    #[test]
    fn test_perturb_pose() {
        let pose = SE3::identity();
        let perturbed = perturb_pose(&pose, 0.05, 2.0, 123);

        // Pose should be different
        let delta = pose.minus(&perturbed, None, None);
        let norm = (delta.rho().norm_squared() + delta.theta().norm_squared()).sqrt();
        assert!(norm > 0.0, "Perturbed pose should differ from original");
        assert!(norm < 0.5, "Perturbation should be small: {}", norm);
    }

    #[test]
    fn test_perturb_landmarks() {
        let landmarks = vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0)];
        let perturbed = perturb_landmarks(&landmarks, 0.03, 456);

        assert_eq!(perturbed.len(), landmarks.len());

        // Each landmark should be slightly different
        for (orig, pert) in landmarks.iter().zip(perturbed.iter()) {
            let diff = (orig - pert).norm();
            assert!(diff > 0.0, "Perturbed landmark should differ");
            assert!(diff < 0.3, "Perturbation should be small: {}", diff);
        }
    }

    #[test]
    fn test_perturb_intrinsics() {
        let intrinsics = vec![500.0, 500.0, 320.0, 240.0];
        let perturbed = perturb_intrinsics(&intrinsics, 0.05, 789);

        assert_eq!(perturbed.len(), intrinsics.len());

        // Each parameter should be within ~5% of original (with high probability)
        for (orig, pert) in intrinsics.iter().zip(perturbed.iter()) {
            let relative_error = (pert - orig).abs() / orig;
            assert!(
                relative_error < 0.2,
                "Perturbation should be reasonable: {:.2}%",
                relative_error * 100.0
            );
        }
    }

    #[test]
    fn test_flatten_unflatten_landmarks() {
        let landmarks = vec![
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::new(7.0, 8.0, 9.0),
        ];

        let flat = flatten_landmarks(&landmarks);
        assert_eq!(flat.len(), 9);
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[1], 2.0);
        assert_eq!(flat[2], 3.0);
        assert_eq!(flat[6], 7.0);

        let unflat = unflatten_landmarks(&flat);
        assert_eq!(unflat.len(), landmarks.len());
        for (orig, restored) in landmarks.iter().zip(unflat.iter()) {
            assert_eq!(orig, restored);
        }
    }
}
