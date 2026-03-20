//! Conversion utilities for Rerun visualization types.
//!
//! This module provides clean conversions from apex-solver's manifold types
//! (SE2, SE3, SO2, SO3, Rn) to Rerun's visualization types. These conversions
//! enable seamless integration with Rerun's real-time visualization system.
//!
//! # Design Philosophy
//!
//! We use extension traits to provide ergonomic conversion methods while
//! respecting Rust's orphan rule. This approach provides:
//! - Clear, self-documenting method names
//! - Better IDE autocomplete and discoverability
//! - Type-safe conversions with zero runtime overhead
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
//! use apex_solver::observers::RerunConvert3D;
//!
//! let pose = SE3::identity();
//! let transform = pose.to_rerun_transform();
//! # }
//! ```
//!
//! ## Batch Conversion
//!
//! ```no_run
//! # #[cfg(feature = "visualization")]
//! # {
//! use apex_solver::manifold::se3::SE3;
//! use apex_solver::observers::CollectRerun3D;
//!
//! let poses = vec![SE3::identity(), SE3::identity()];
//! let points = poses.iter().collect_points3d();
//! # }
//! ```

#[cfg(feature = "visualization")]
use apex_manifolds::{se2::SE2, se3::SE3};
#[cfg(feature = "visualization")]
use rerun::{Arrows2D, Arrows3D, Points2D, Points3D, Transform3D, Vec2D, Vec3D};

// ============================================================================
// Consolidated Rerun conversion traits
// ============================================================================

/// Rerun conversion methods for SE3 types.
///
/// Provides ergonomic conversion from SE3 poses to all Rerun 3D primitive types.
#[cfg(feature = "visualization")]
pub trait RerunConvert3D {
    /// Convert this SE3 pose to a Rerun Transform3D (translation + rotation).
    fn to_rerun_transform(&self) -> Transform3D;

    /// Extract the translation component as a Rerun Vec3D.
    fn to_rerun_vec3d(&self) -> Vec3D;

    /// Convert this SE3 pose to a single point in Rerun Points3D format.
    fn to_rerun_points3d(&self) -> Points3D;

    /// Convert this SE3 pose to coordinate frame arrows in Rerun.
    ///
    /// Creates three arrows (X=red, Y=green, Z=blue) showing the pose's
    /// orientation, rooted at the pose's translation.
    fn to_rerun_arrows3d(&self) -> Arrows3D;
}

/// Rerun conversion methods for SE2 types.
///
/// Provides ergonomic conversion from SE2 poses to all Rerun 2D primitive types.
#[cfg(feature = "visualization")]
pub trait RerunConvert2D {
    /// Extract the 2D translation component as a Rerun Vec2D.
    fn to_rerun_vec2d(&self) -> Vec2D;

    /// Convert this SE2 pose to a single point in Rerun Points2D format.
    fn to_rerun_points2d(&self) -> Points2D;

    /// Convert this SE2 pose to coordinate frame arrows in Rerun.
    ///
    /// Creates two arrows (X=red, Y=green) showing the pose's orientation,
    /// rooted at the pose's translation.
    fn to_rerun_arrows2d(&self) -> Arrows2D;

    /// Convert this SE2 pose to a 3D transform at the z=0 plane.
    ///
    /// Useful for visualizing 2D poses in a 3D viewer.
    fn to_rerun_transform_3d(&self) -> Transform3D;
}

/// Batch Rerun collection for SE3 iterators.
#[cfg(feature = "visualization")]
pub trait CollectRerun3D<'a> {
    /// Collect SE3 poses into a Points3D cloud (translation components only).
    fn collect_points3d(self) -> Points3D;

    /// Collect SE3 poses into coordinate frame arrows.
    ///
    /// Creates three arrows (X=red, Y=green, Z=blue) for each pose.
    fn collect_arrows3d(self) -> Arrows3D;
}

/// Batch Rerun collection for SE2 iterators.
#[cfg(feature = "visualization")]
pub trait CollectRerun2D<'a> {
    /// Collect SE2 poses into a Points2D cloud (translation components only).
    fn collect_points2d(self) -> Points2D;

    /// Collect SE2 poses into coordinate frame arrows.
    ///
    /// Creates two arrows (X=red, Y=green) for each pose.
    fn collect_arrows2d(self) -> Arrows2D;
}

// ============================================================================
// SE3 Implementation
// ============================================================================

#[cfg(feature = "visualization")]
impl RerunConvert3D for SE3 {
    fn to_rerun_transform(&self) -> Transform3D {
        let trans = self.translation();
        let rot = self.rotation_quaternion();

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

    fn to_rerun_vec3d(&self) -> Vec3D {
        let trans = self.translation();
        Vec3D::new(trans.x as f32, trans.y as f32, trans.z as f32)
    }

    fn to_rerun_points3d(&self) -> Points3D {
        let vec = self.to_rerun_vec3d();
        Points3D::new([vec])
    }

    fn to_rerun_arrows3d(&self) -> Arrows3D {
        let rot_quat = self.rotation_quaternion();
        let rot_mat = rot_quat.to_rotation_matrix();
        let trans = self.translation();

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

// ============================================================================
// SE2 Implementation
// ============================================================================

#[cfg(feature = "visualization")]
impl RerunConvert2D for SE2 {
    fn to_rerun_vec2d(&self) -> Vec2D {
        Vec2D::new(self.x() as f32, self.y() as f32)
    }

    fn to_rerun_points2d(&self) -> Points2D {
        let vec = self.to_rerun_vec2d();
        Points2D::new([vec])
    }

    fn to_rerun_arrows2d(&self) -> Arrows2D {
        let rot_mat = self.rotation_matrix();

        let x_axis = [rot_mat[(0, 0)] as f32, rot_mat[(1, 0)] as f32];
        let y_axis = [rot_mat[(0, 1)] as f32, rot_mat[(1, 1)] as f32];

        let origin = [self.x() as f32, self.y() as f32];

        Arrows2D::from_vectors([x_axis, y_axis])
            .with_origins([origin, origin])
            .with_colors([[255, 0, 0], [0, 255, 0]]) // Red/Green for X/Y
    }

    fn to_rerun_transform_3d(&self) -> Transform3D {
        let position = rerun::external::glam::Vec3::new(self.x() as f32, self.y() as f32, 0.0);

        let angle = self.angle();
        let half_angle = (angle / 2.0) as f32;
        let rotation =
            rerun::external::glam::Quat::from_xyzw(0.0, 0.0, half_angle.sin(), half_angle.cos());

        Transform3D::from_translation_rotation(position, rotation)
    }
}

// ============================================================================
// Batch SE3 Iterator Implementations
// ============================================================================

#[cfg(feature = "visualization")]
impl<'a, I> CollectRerun3D<'a> for I
where
    I: Iterator<Item = &'a SE3>,
{
    fn collect_points3d(self) -> Points3D {
        let points: Vec<Vec3D> = self.map(|se3| se3.to_rerun_vec3d()).collect();
        Points3D::new(points)
    }

    fn collect_arrows3d(self) -> Arrows3D {
        let mut vectors = Vec::new();
        let mut origins = Vec::new();
        let mut colors = Vec::new();

        for se3 in self {
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
// Batch SE2 Iterator Implementations
// ============================================================================

#[cfg(feature = "visualization")]
impl<'a, I> CollectRerun2D<'a> for I
where
    I: Iterator<Item = &'a SE2>,
{
    fn collect_points2d(self) -> Points2D {
        let points: Vec<Vec2D> = self.map(|se2| se2.to_rerun_vec2d()).collect();
        Points2D::new(points)
    }

    fn collect_arrows2d(self) -> Arrows2D {
        let mut vectors = Vec::new();
        let mut origins = Vec::new();
        let mut colors = Vec::new();

        for se2 in self {
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
        let vec = pose.to_rerun_vec3d();

        assert_eq!(vec.x(), 0.0);
        assert_eq!(vec.y(), 0.0);
        assert_eq!(vec.z(), 0.0);
    }

    #[test]
    fn test_se2_to_vec2d() {
        use apex_manifolds::se2::SE2;

        let pose = SE2::identity();
        let vec = pose.to_rerun_vec2d();

        assert_eq!(vec.x(), 0.0);
        assert_eq!(vec.y(), 0.0);
    }

    #[test]
    fn test_se3_collection_to_points() {
        use apex_manifolds::se3::SE3;

        let poses = [SE3::identity(), SE3::identity(), SE3::identity()];
        let points = poses.iter().collect_points3d();

        let _ = points;
    }

    #[test]
    fn test_se2_collection_to_arrows() {
        use apex_manifolds::se2::SE2;

        let poses = [SE2::identity(), SE2::identity()];
        let arrows = poses.iter().collect_arrows2d();

        let _ = arrows;
    }
}
