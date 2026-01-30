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
//! use apex_solver::observers::ToRerunTransform3D;
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
//! use apex_solver::observers::CollectRerunPoints3D;
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
// Extension Traits for SE3 → Rerun Conversions
// ============================================================================

/// Convert SE3 to Rerun Transform3D (translation + rotation).
#[cfg(feature = "visualization")]
pub trait ToRerunTransform3D {
    /// Convert this SE3 pose to a Rerun Transform3D.
    ///
    /// # Returns
    /// Transform3D with translation and rotation from the SE3 pose.
    fn to_rerun_transform(&self) -> Transform3D;
}

/// Convert SE3 to Rerun Vec3D (translation only).
#[cfg(feature = "visualization")]
pub trait ToRerunVec3D {
    /// Extract the translation component as a Rerun Vec3D.
    ///
    /// # Returns
    /// Vec3D containing only the translation (x, y, z) of the SE3 pose.
    fn to_rerun_vec3d(&self) -> Vec3D;
}

/// Convert SE3 to Rerun Points3D (single point at translation).
#[cfg(feature = "visualization")]
pub trait ToRerunPoints3D {
    /// Convert this SE3 pose to a single point in Rerun Points3D format.
    ///
    /// # Returns
    /// Points3D containing one point at the pose's translation.
    fn to_rerun_points3d(&self) -> Points3D;
}

/// Convert SE3 to Rerun Arrows3D (pose with orientation axes).
#[cfg(feature = "visualization")]
pub trait ToRerunArrows3D {
    /// Convert this SE3 pose to coordinate frame arrows in Rerun.
    ///
    /// Creates three arrows (X=red, Y=green, Z=blue) showing the pose's
    /// orientation, rooted at the pose's translation.
    ///
    /// # Returns
    /// Arrows3D showing the three basis vectors of the pose's rotation.
    fn to_rerun_arrows3d(&self) -> Arrows3D;
}

// ============================================================================
// Extension Traits for SE2 → Rerun Conversions
// ============================================================================

/// Convert SE2 to Rerun Vec2D (translation only).
#[cfg(feature = "visualization")]
pub trait ToRerunVec2D {
    /// Extract the 2D translation component as a Rerun Vec2D.
    ///
    /// # Returns
    /// Vec2D containing the (x, y) translation of the SE2 pose.
    fn to_rerun_vec2d(&self) -> Vec2D;
}

/// Convert SE2 to Rerun Points2D (single point at translation).
#[cfg(feature = "visualization")]
pub trait ToRerunPoints2D {
    /// Convert this SE2 pose to a single point in Rerun Points2D format.
    ///
    /// # Returns
    /// Points2D containing one point at the pose's translation.
    fn to_rerun_points2d(&self) -> Points2D;
}

/// Convert SE2 to Rerun Arrows2D (pose with orientation axes).
#[cfg(feature = "visualization")]
pub trait ToRerunArrows2D {
    /// Convert this SE2 pose to coordinate frame arrows in Rerun.
    ///
    /// Creates two arrows (X=red, Y=green) showing the pose's orientation,
    /// rooted at the pose's translation.
    ///
    /// # Returns
    /// Arrows2D showing the two basis vectors of the pose's rotation.
    fn to_rerun_arrows2d(&self) -> Arrows2D;
}

/// Convert SE2 to Rerun Transform3D (places SE2 at z=0 plane).
#[cfg(feature = "visualization")]
pub trait ToRerunTransform3DFrom2D {
    /// Convert this SE2 pose to a 3D transform at the z=0 plane.
    ///
    /// Useful for visualizing 2D poses in a 3D viewer. The rotation is
    /// converted to a rotation around the Z-axis.
    ///
    /// # Returns
    /// Transform3D with (x, y, 0) translation and rotation around Z-axis.
    fn to_rerun_transform_3d(&self) -> Transform3D;
}

// ============================================================================
// Extension Traits for Batch Conversions (Iterator Extensions)
// ============================================================================

/// Collect an iterator of SE3 poses into Rerun Points3D.
#[cfg(feature = "visualization")]
pub trait CollectRerunPoints3D<'a> {
    /// Collect SE3 poses into a Points3D cloud (translation components only).
    ///
    /// # Returns
    /// Points3D containing one point for each SE3 pose in the iterator.
    fn collect_points3d(self) -> Points3D;
}

/// Collect an iterator of SE3 poses into Rerun Arrows3D.
#[cfg(feature = "visualization")]
pub trait CollectRerunArrows3D<'a> {
    /// Collect SE3 poses into coordinate frame arrows.
    ///
    /// Creates three arrows (X=red, Y=green, Z=blue) for each pose.
    ///
    /// # Returns
    /// Arrows3D showing orientation axes for all poses in the iterator.
    fn collect_arrows3d(self) -> Arrows3D;
}

/// Collect an iterator of SE2 poses into Rerun Points2D.
#[cfg(feature = "visualization")]
pub trait CollectRerunPoints2D<'a> {
    /// Collect SE2 poses into a Points2D cloud (translation components only).
    ///
    /// # Returns
    /// Points2D containing one point for each SE2 pose in the iterator.
    fn collect_points2d(self) -> Points2D;
}

/// Collect an iterator of SE2 poses into Rerun Arrows2D.
#[cfg(feature = "visualization")]
pub trait CollectRerunArrows2D<'a> {
    /// Collect SE2 poses into coordinate frame arrows.
    ///
    /// Creates two arrows (X=red, Y=green) for each pose.
    ///
    /// # Returns
    /// Arrows2D showing orientation axes for all poses in the iterator.
    fn collect_arrows2d(self) -> Arrows2D;
}

// ============================================================================
// SE3 Conversions Implementation
// ============================================================================

#[cfg(feature = "visualization")]
impl ToRerunTransform3D for SE3 {
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
}

#[cfg(feature = "visualization")]
impl ToRerunVec3D for SE3 {
    fn to_rerun_vec3d(&self) -> Vec3D {
        let trans = self.translation();
        Vec3D::new(trans.x as f32, trans.y as f32, trans.z as f32)
    }
}

#[cfg(feature = "visualization")]
impl ToRerunPoints3D for SE3 {
    fn to_rerun_points3d(&self) -> Points3D {
        let vec = self.to_rerun_vec3d();
        Points3D::new([vec])
    }
}

#[cfg(feature = "visualization")]
impl ToRerunArrows3D for SE3 {
    fn to_rerun_arrows3d(&self) -> Arrows3D {
        let rot_quat = self.rotation_quaternion();
        let rot_mat = rot_quat.to_rotation_matrix();
        let trans = self.translation();

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
impl<'a, I> CollectRerunPoints3D<'a> for I
where
    I: Iterator<Item = &'a SE3>,
{
    fn collect_points3d(self) -> Points3D {
        let points: Vec<Vec3D> = self.map(|se3| se3.to_rerun_vec3d()).collect();
        Points3D::new(points)
    }
}

/// Collect multiple SE3 poses into Arrows3D (translation + orientation axes)
#[cfg(feature = "visualization")]
impl<'a, I> CollectRerunArrows3D<'a> for I
where
    I: Iterator<Item = &'a SE3>,
{
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
// SE2 Conversions Implementation
// ============================================================================

#[cfg(feature = "visualization")]
impl ToRerunVec2D for SE2 {
    fn to_rerun_vec2d(&self) -> Vec2D {
        Vec2D::new(self.x() as f32, self.y() as f32)
    }
}

#[cfg(feature = "visualization")]
impl ToRerunPoints2D for SE2 {
    fn to_rerun_points2d(&self) -> Points2D {
        let vec = self.to_rerun_vec2d();
        Points2D::new([vec])
    }
}

#[cfg(feature = "visualization")]
impl ToRerunArrows2D for SE2 {
    fn to_rerun_arrows2d(&self) -> Arrows2D {
        let rot_mat = self.rotation_matrix();

        let x_axis = [rot_mat[(0, 0)] as f32, rot_mat[(1, 0)] as f32];
        let y_axis = [rot_mat[(0, 1)] as f32, rot_mat[(1, 1)] as f32];

        let origin = [self.x() as f32, self.y() as f32];

        Arrows2D::from_vectors([x_axis, y_axis])
            .with_origins([origin, origin])
            .with_colors([[255, 0, 0], [0, 255, 0]]) // Red/Green for X/Y
    }
}

/// Convert SE2 to 3D transform (places at z=0 plane)
#[cfg(feature = "visualization")]
impl ToRerunTransform3DFrom2D for SE2 {
    fn to_rerun_transform_3d(&self) -> Transform3D {
        let position = rerun::external::glam::Vec3::new(self.x() as f32, self.y() as f32, 0.0);

        // Create quaternion from 2D rotation (rotation around Z-axis)
        let angle = self.angle();
        let half_angle = (angle / 2.0) as f32;
        let rotation =
            rerun::external::glam::Quat::from_xyzw(0.0, 0.0, half_angle.sin(), half_angle.cos());

        Transform3D::from_translation_rotation(position, rotation)
    }
}

/// Collect multiple SE2 poses into Points2D
#[cfg(feature = "visualization")]
impl<'a, I> CollectRerunPoints2D<'a> for I
where
    I: Iterator<Item = &'a SE2>,
{
    fn collect_points2d(self) -> Points2D {
        let points: Vec<Vec2D> = self.map(|se2| se2.to_rerun_vec2d()).collect();
        Points2D::new(points)
    }
}

/// Collect multiple SE2 poses into Arrows2D (position + orientation)
#[cfg(feature = "visualization")]
impl<'a, I> CollectRerunArrows2D<'a> for I
where
    I: Iterator<Item = &'a SE2>,
{
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

        // Should create Points3D with 3 points
        // (Rerun API doesn't expose count directly, so just verify it compiles)
        let _ = points;
    }

    #[test]
    fn test_se2_collection_to_arrows() {
        use apex_manifolds::se2::SE2;

        let poses = [SE2::identity(), SE2::identity()];
        let arrows = poses.iter().collect_arrows2d();

        // Should create Arrows2D with orientation vectors
        let _ = arrows;
    }
}
