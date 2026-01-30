//! Conversion utilities for Rerun visualization types.
//!
//! This module provides clean conversions from apex-solver's manifold types
//! (SE2, SE3, SO2, SO3, Rn) to Rerun's visualization types. These conversions
//! enable seamless integration with Rerun's real-time visualization system.
//!
//! # Design Philosophy
//!
//! Following the fact-rs pattern, we provide:
//! - Type conversions via `From` trait for single values
//! - Batch conversions via `FromIterator` for collections
//! - Zero-copy conversions where possible
//! - Consistent color schemes (red/green/blue for x/y/z axes)
//!
//! # Examples
//!
//! ## Single Pose Conversion
//!
//! ```no_run
//! # #[cfg(feature = "visualization")]
//! # {
//! use apex_solver::manifold::se3::SE3;
//! use rerun::Transform3D;
//!
//! let pose = SE3::identity();
//! let transform: Transform3D = (&pose).into();
//! # }
//! ```
//!
//! ## Batch Conversion
//!
//! ```no_run
//! # #[cfg(feature = "visualization")]
//! # {
//! use apex_solver::manifold::se3::SE3;
//! use rerun::Points3D;
//!
//! let poses = vec![SE3::identity(), SE3::identity()];
//! let points: Points3D = poses.iter().collect();
//! # }
//! ```

#[cfg(feature = "visualization")]
use apex_manifolds::{se2::SE2, se3::SE3};
#[cfg(feature = "visualization")]
use rerun::{Arrows2D, Arrows3D, Points2D, Points3D, Transform3D, Vec2D, Vec3D};

// ============================================================================
// SE3 Conversions (3D Rigid Body Transforms)
// ============================================================================

#[cfg(feature = "visualization")]
impl From<&SE3> for Transform3D {
    fn from(se3: &SE3) -> Transform3D {
        let trans = se3.translation();
        let rot = se3.rotation_quaternion();

        let position =
            rerun::external::glam::Vec3::new(trans.x as f32, trans.y as f32, trans.z as f32);

        let rotation = rerun::external::glam::Quat::from_xyzw(
            rot.as_ref().i as f32,
            rot.as_ref().j as f32,
            rot.as_ref().k as f32,
            rot.as_ref().w as f32,
        );

        Transform3D::from_translation_rotation(position, rotation)
    }
}

#[cfg(feature = "visualization")]
impl From<&SE3> for Vec3D {
    fn from(se3: &SE3) -> Vec3D {
        let trans = se3.translation();
        Vec3D::new(trans.x as f32, trans.y as f32, trans.z as f32)
    }
}

#[cfg(feature = "visualization")]
impl From<&SE3> for Points3D {
    fn from(se3: &SE3) -> Points3D {
        let vec: Vec3D = se3.into();
        Points3D::new([vec])
    }
}

#[cfg(feature = "visualization")]
impl From<&SE3> for Arrows3D {
    fn from(se3: &SE3) -> Arrows3D {
        let rot_quat = se3.rotation_quaternion();
        let rot_mat = rot_quat.to_rotation_matrix();
        let trans = se3.translation();

        // Extract basis vectors (columns of rotation matrix)
        let x_axis = [
            rot_mat[(0, 0)] as f32,
            rot_mat[(1, 0)] as f32,
            rot_mat[(2, 0)] as f32,
        ];
        let y_axis = [
            rot_mat[(0, 1)] as f32,
            rot_mat[(1, 1)] as f32,
            rot_mat[(2, 1)] as f32,
        ];
        let z_axis = [
            rot_mat[(0, 2)] as f32,
            rot_mat[(1, 2)] as f32,
            rot_mat[(2, 2)] as f32,
        ];

        let origin = [trans.x as f32, trans.y as f32, trans.z as f32];

        Arrows3D::from_vectors([x_axis, y_axis, z_axis])
            .with_origins([origin, origin, origin])
            .with_colors([[255, 0, 0], [0, 255, 0], [0, 0, 255]]) // RGB for XYZ
    }
}

/// Collect multiple SE3 poses into a Points3D (just translation components)
#[cfg(feature = "visualization")]
impl<'a> FromIterator<&'a SE3> for Points3D {
    fn from_iter<I: IntoIterator<Item = &'a SE3>>(iter: I) -> Points3D {
        let points: Vec<Vec3D> = iter.into_iter().map(|se3| se3.into()).collect();
        Points3D::new(points)
    }
}

/// Collect multiple SE3 poses into Arrows3D (translation + orientation axes)
#[cfg(feature = "visualization")]
impl<'a> FromIterator<&'a SE3> for Arrows3D {
    fn from_iter<I: IntoIterator<Item = &'a SE3>>(iter: I) -> Arrows3D {
        let mut vectors = Vec::new();
        let mut origins = Vec::new();
        let mut colors = Vec::new();

        for se3 in iter {
            let rot_quat = se3.rotation_quaternion();
            let rot_mat = rot_quat.to_rotation_matrix();
            let trans = se3.translation();

            let x_axis = [
                rot_mat[(0, 0)] as f32,
                rot_mat[(1, 0)] as f32,
                rot_mat[(2, 0)] as f32,
            ];
            let y_axis = [
                rot_mat[(0, 1)] as f32,
                rot_mat[(1, 1)] as f32,
                rot_mat[(2, 1)] as f32,
            ];
            let z_axis = [
                rot_mat[(0, 2)] as f32,
                rot_mat[(1, 2)] as f32,
                rot_mat[(2, 2)] as f32,
            ];

            let origin = [trans.x as f32, trans.y as f32, trans.z as f32];

            vectors.push(x_axis);
            vectors.push(y_axis);
            vectors.push(z_axis);
            origins.push(origin);
            origins.push(origin);
            origins.push(origin);
            colors.push([255, 0, 0]); // X = red
            colors.push([0, 255, 0]); // Y = green
            colors.push([0, 0, 255]); // Z = blue
        }

        Arrows3D::from_vectors(vectors)
            .with_origins(origins)
            .with_colors(colors)
    }
}

// ============================================================================
// SE2 Conversions (2D Rigid Body Transforms)
// ============================================================================

#[cfg(feature = "visualization")]
impl From<&SE2> for Vec2D {
    fn from(se2: &SE2) -> Vec2D {
        Vec2D::new(se2.x() as f32, se2.y() as f32)
    }
}

#[cfg(feature = "visualization")]
impl From<&SE2> for Points2D {
    fn from(se2: &SE2) -> Points2D {
        let vec: Vec2D = se2.into();
        Points2D::new([vec])
    }
}

#[cfg(feature = "visualization")]
impl From<&SE2> for Arrows2D {
    fn from(se2: &SE2) -> Arrows2D {
        let rot_mat = se2.rotation_matrix();

        let x_axis = [rot_mat[(0, 0)] as f32, rot_mat[(1, 0)] as f32];
        let y_axis = [rot_mat[(0, 1)] as f32, rot_mat[(1, 1)] as f32];

        let origin = [se2.x() as f32, se2.y() as f32];

        Arrows2D::from_vectors([x_axis, y_axis])
            .with_origins([origin, origin])
            .with_colors([[255, 0, 0], [0, 255, 0]]) // Red/Green for X/Y
    }
}

/// Convert SE2 to 3D transform (places at z=0 plane)
#[cfg(feature = "visualization")]
impl From<&SE2> for Transform3D {
    fn from(se2: &SE2) -> Transform3D {
        let position = rerun::external::glam::Vec3::new(se2.x() as f32, se2.y() as f32, 0.0);

        // Create quaternion from 2D rotation (rotation around Z-axis)
        let angle = se2.angle();
        let half_angle = (angle / 2.0) as f32;
        let rotation =
            rerun::external::glam::Quat::from_xyzw(0.0, 0.0, half_angle.sin(), half_angle.cos());

        Transform3D::from_translation_rotation(position, rotation)
    }
}

/// Collect multiple SE2 poses into Points2D
#[cfg(feature = "visualization")]
impl<'a> FromIterator<&'a SE2> for Points2D {
    fn from_iter<I: IntoIterator<Item = &'a SE2>>(iter: I) -> Points2D {
        let points: Vec<Vec2D> = iter.into_iter().map(|se2| se2.into()).collect();
        Points2D::new(points)
    }
}

/// Collect multiple SE2 poses into Arrows2D (position + orientation)
#[cfg(feature = "visualization")]
impl<'a> FromIterator<&'a SE2> for Arrows2D {
    fn from_iter<I: IntoIterator<Item = &'a SE2>>(iter: I) -> Arrows2D {
        let mut vectors = Vec::new();
        let mut origins = Vec::new();
        let mut colors = Vec::new();

        for se2 in iter {
            let rot_mat = se2.rotation_matrix();

            let x_axis = [rot_mat[(0, 0)] as f32, rot_mat[(1, 0)] as f32];
            let y_axis = [rot_mat[(0, 1)] as f32, rot_mat[(1, 1)] as f32];

            let origin = [se2.x() as f32, se2.y() as f32];

            vectors.push(x_axis);
            vectors.push(y_axis);
            origins.push(origin);
            origins.push(origin);
            colors.push([255, 0, 0]); // X = red
            colors.push([0, 255, 0]); // Y = green
        }

        Arrows2D::from_vectors(vectors)
            .with_origins(origins)
            .with_colors(colors)
    }
}

#[cfg(test)]
#[cfg(feature = "visualization")]
mod tests {
    use super::*;

    #[test]
    fn test_se3_to_vec3d() {
        use apex_manifolds::se3::SE3;

        let pose = SE3::identity();
        let vec: Vec3D = (&pose).into();

        assert_eq!(vec.x(), 0.0);
        assert_eq!(vec.y(), 0.0);
        assert_eq!(vec.z(), 0.0);
    }

    #[test]
    fn test_se2_to_vec2d() {
        use apex_manifolds::se2::SE2;

        let pose = SE2::identity();
        let vec: Vec2D = (&pose).into();

        assert_eq!(vec.x(), 0.0);
        assert_eq!(vec.y(), 0.0);
    }

    #[test]
    fn test_se3_collection_to_points() {
        use apex_manifolds::se3::SE3;

        let poses = [SE3::identity(), SE3::identity(), SE3::identity()];
        let points: Points3D = poses.iter().collect();

        // Should create Points3D with 3 points
        // (Rerun API doesn't expose count directly, so just verify it compiles)
        let _ = points;
    }

    #[test]
    fn test_se2_collection_to_arrows() {
        use apex_manifolds::se2::SE2;

        let poses = [SE2::identity(), SE2::identity()];
        let arrows: Arrows2D = poses.iter().collect();

        // Should create Arrows2D with orientation vectors
        let _ = arrows;
    }
}
