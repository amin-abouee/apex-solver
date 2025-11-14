//! Real-time optimization visualization using Rerun.
//!
//! This module provides comprehensive visualization of the optimization process,
//! including time series plots, matrix visualizations, and manifold state tracking.
//!
//! # Features
//!
//! - **Time series plots**: Cost, gradient norm, damping parameter, step quality
//! - **Sparse Hessian visualization**: Heat map showing matrix structure and values
//! - **Gradient visualization**: Vector representation with magnitude encoding
//! - **Manifold state**: Real-time pose updates for SE2/SE3 problems
//!
//! # Feature Flag
//!
//! This module requires the `visualization` feature to be enabled. To use visualization,
//! enable the feature in your `Cargo.toml`:
//!
//! ```toml
//! apex-solver = { version = "0.1", features = ["visualization"] }
//! ```

use crate::{core::problem, io};
use faer::sparse;
use std::collections;
use tracing::warn;

/// Optimization visualizer for real-time debugging with Rerun.
///
/// This struct manages all visualization state and provides methods for logging
/// various aspects of the optimization process.
#[derive(Debug)]
pub struct OptimizationVisualizer {
    rec: Option<rerun::RecordingStream>,
    enabled: bool,
}

impl OptimizationVisualizer {
    /// Create a new optimization visualizer.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable visualization
    ///
    /// # Returns
    ///
    /// A new visualizer instance.
    pub fn new(enabled: bool) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_options(enabled, None)
    }

    /// Create a new optimization visualizer with file save option.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable visualization
    /// * `save_path` - Optional path to save recording to file instead of spawning viewer
    pub fn new_with_options(
        enabled: bool,
        save_path: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let rec = if enabled {
            let rec = if let Some(path) = save_path {
                // Save to file
                println!("✓ Saving visualization to: {}", path);
                rerun::RecordingStreamBuilder::new("apex-solver-optimization").save(path)?
            } else {
                // Try to spawn Rerun viewer
                match rerun::RecordingStreamBuilder::new("apex-solver-optimization").spawn() {
                    Ok(rec) => {
                        println!("✓ Rerun viewer launched successfully");
                        rec
                    }
                    Err(e) => {
                        warn!("Could not launch Rerun viewer: {}", e);
                        warn!("Saving to file 'optimization.rrd' instead");
                        warn!("View it later with: rerun optimization.rrd");

                        // Fall back to saving to file
                        rerun::RecordingStreamBuilder::new("apex-solver-optimization")
                            .save("optimization.rrd")?
                    }
                }
            };

            Some(rec)
        } else {
            None
        };

        Ok(Self { rec, enabled })
    }

    /// Check if visualization is enabled and active.
    #[inline(always)]
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.rec.is_some()
    }

    /// Log scalar values for time series plots.
    ///
    /// Each metric is logged to a separate entity path for independent scaling.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Current iteration number
    /// * `cost` - Current cost value
    /// * `gradient_norm` - L2 norm of the gradient
    /// * `damping` - Current damping parameter (λ)
    /// * `step_norm` - L2 norm of the parameter update
    /// * `step_quality` - Step quality metric (ρ)
    pub fn log_scalars(
        &self,
        iteration: usize,
        cost: f64,
        gradient_norm: f64,
        damping: f64,
        step_norm: f64,
        step_quality: Option<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_enabled() {
            return Ok(());
        }

        let rec = self.rec.as_ref().unwrap();

        // Set timeline
        rec.set_time_sequence("iteration", iteration as i64);

        // Log each metric to COMPLETELY SEPARATE top-level spaces
        // Rerun automatically creates separate TimeSeriesView for each top-level space
        // Using different prefixes (cost_plot, gradient_plot, etc.) ensures separate panels
        rec.log("cost_plot/value", &rerun::archetypes::Scalars::new([cost]))?;

        rec.log(
            "gradient_plot/norm",
            &rerun::archetypes::Scalars::new([gradient_norm]),
        )?;

        rec.log(
            "damping_plot/lambda",
            &rerun::archetypes::Scalars::new([damping]),
        )?;

        rec.log(
            "step_plot/norm",
            &rerun::archetypes::Scalars::new([step_norm]),
        )?;

        if let Some(rho) = step_quality {
            rec.log("quality_plot/rho", &rerun::archetypes::Scalars::new([rho]))?;
        }

        Ok(())
    }

    /// Log iteration timing information.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Current iteration number
    /// * `duration_ms` - Time taken for this iteration in milliseconds
    pub fn log_timing(
        &self,
        iteration: usize,
        duration_ms: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_enabled() {
            return Ok(());
        }

        let rec = self.rec.as_ref().unwrap();
        rec.set_time_sequence("iteration", iteration as i64);

        rec.log(
            "timing_plot/iteration_ms",
            &rerun::archetypes::Scalars::new([duration_ms]),
        )?;

        Ok(())
    }

    /// Log sparse Hessian matrix as a heat map image.
    ///
    /// The Hessian is downsampled to a fixed 100×100 image for visualization.
    /// Uses a blue-white-red color scheme:
    /// - Negative values: blue
    /// - Zero values: white
    /// - Positive values: red
    ///
    /// # Arguments
    ///
    /// * `hessian` - The sparse Hessian matrix
    /// * `iteration` - Current iteration number
    pub fn log_hessian(
        &self,
        hessian: Option<&sparse::SparseColMat<usize, f64>>,
        iteration: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_enabled() {
            return Ok(());
        }

        let Some(hessian) = hessian else {
            return Ok(());
        };

        let rec = self.rec.as_ref().unwrap();
        rec.set_time_sequence("iteration", iteration as i64);

        // Convert sparse Hessian to fixed 100×100 image
        let image_data = Self::sparse_hessian_to_image(hessian)?;

        // Log as tensor (Rerun will display it as an image)
        rec.log(
            "optimization/matrices/hessian",
            &rerun::archetypes::Tensor::new(image_data),
        )?;

        Ok(())
    }

    /// Log gradient vector as a tensor visualization.
    ///
    /// The gradient is downsampled to a fixed 100-element vector and displayed
    /// as a horizontal bar image.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient vector
    /// * `iteration` - Current iteration number
    pub fn log_gradient(
        &self,
        gradient: Option<&faer::Mat<f64>>,
        iteration: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_enabled() {
            return Ok(());
        }

        let Some(grad) = gradient else {
            return Ok(());
        };

        let rec = self.rec.as_ref().unwrap();
        rec.set_time_sequence("iteration", iteration as i64);

        // Convert gradient to 1D heat map
        let grad_vec: Vec<f64> = (0..grad.nrows()).map(|i| grad[(i, 0)]).collect();

        // Create a horizontal bar visualization as an image (fixed 100 width)
        let image_data = Self::gradient_to_image(&grad_vec)?;

        rec.log(
            "optimization/matrices/gradient",
            &rerun::archetypes::Tensor::new(image_data),
        )?;

        Ok(())
    }

    /// Log current manifold states for SE2/SE3 problems.
    ///
    /// This visualizes the current pose estimates during optimization.
    /// Only the latest poses are shown (previous iterations are cleared).
    ///
    /// # Arguments
    ///
    /// * `variables` - Current variable values
    /// * `iteration` - Current iteration number
    pub fn log_manifolds(
        &self,
        variables: &collections::HashMap<String, problem::VariableEnum>,
        iteration: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_enabled() {
            return Ok(());
        }

        let rec = self.rec.as_ref().unwrap();
        rec.set_time_sequence("iteration", iteration as i64);

        // Visualize SE3 poses (only poses, no edges)
        for (var_name, var) in variables {
            match var {
                problem::VariableEnum::SE3(v) => {
                    let trans = v.value.translation();
                    let rot = v.value.rotation_quaternion();

                    // Convert to glam types for Rerun
                    let position = rerun::external::glam::Vec3::new(
                        trans.x as f32,
                        trans.y as f32,
                        trans.z as f32,
                    );

                    let nq = rot.as_ref();
                    let rotation = rerun::external::glam::Quat::from_xyzw(
                        nq.i as f32,
                        nq.j as f32,
                        nq.k as f32,
                        nq.w as f32,
                    );

                    let transform =
                        rerun::Transform3D::from_translation_rotation(position, rotation);

                    rec.log(
                        format!("optimized_graph/se3_poses/{}", var_name),
                        &transform,
                    )?;

                    // Also log a small pinhole camera for better visualization
                    rec.log(
                        format!("optimized_graph/se3_poses/{}", var_name),
                        &rerun::archetypes::Pinhole::from_fov_and_aspect_ratio(0.5, 1.0),
                    )?;
                }
                problem::VariableEnum::SE2(v) => {
                    // Visualize SE2 as 2D points or 3D poses at z=0
                    let x = v.value.x();
                    let y = v.value.y();

                    let position = rerun::external::glam::Vec3::new(x as f32, y as f32, 0.0);
                    let rotation = rerun::external::glam::Quat::IDENTITY;

                    let transform =
                        rerun::Transform3D::from_translation_rotation(position, rotation);

                    rec.log(
                        format!("optimized_graph/se2_poses/{}", var_name),
                        &transform,
                    )?;
                }
                _ => {
                    // Skip other manifold types (SO2, SO3, Rn)
                }
            }
        }

        Ok(())
    }

    /// Log convergence status and final summary.
    ///
    /// # Arguments
    ///
    /// * `status` - Convergence status message
    pub fn log_convergence(&self, status: &str) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_enabled() {
            return Ok(());
        }

        let rec = self.rec.as_ref().unwrap();

        // Log as a text annotation
        rec.log(
            "optimization/status",
            &rerun::archetypes::TextDocument::new(status),
        )?;

        Ok(())
    }

    /// Log the initial graph structure before optimization.
    ///
    /// This visualizes only the initial poses (vertices), not edges.
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph structure loaded from G2O file
    /// * `scale` - Scale factor for visualization
    pub fn log_initial_graph(
        &self,
        graph: &io::Graph,
        scale: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_enabled() {
            return Ok(());
        }

        let rec = self.rec.as_ref().unwrap();

        // Visualize SE3 vertices only (no edges)
        for (id, vertex) in &graph.vertices_se3 {
            let (position, rotation) = vertex.to_rerun_transform(scale);
            let transform = rerun::Transform3D::from_translation_rotation(position, rotation);

            let entity_path = format!("initial_graph/se3_poses/{}", id);
            rec.log(entity_path.as_str(), &transform)?;

            // Add a small pinhole camera for better visualization
            rec.log(
                entity_path.as_str(),
                &rerun::archetypes::Pinhole::from_fov_and_aspect_ratio(0.5, 1.0),
            )?;
        }

        // Visualize SE2 vertices only (no edges)
        if !graph.vertices_se2.is_empty() {
            let positions: Vec<[f32; 2]> = graph
                .vertices_se2
                .values()
                .map(|vertex| vertex.to_rerun_position_2d(scale))
                .collect();

            let colors = vec![rerun::components::Color::from_rgb(100, 150, 255); positions.len()];

            rec.log(
                "initial_graph/se2_poses",
                &rerun::archetypes::Points2D::new(positions)
                    .with_colors(colors)
                    .with_radii([0.5 * scale]),
            )?;
        }

        Ok(())
    }

    // ========================================================================
    // Private helper methods for image conversions
    // ========================================================================

    /// Convert sparse Hessian matrix to fixed 100×100 RGB image with heat map coloring.
    ///
    /// Uses block averaging to downsample large matrices while preserving structure.
    fn sparse_hessian_to_image(
        hessian: &sparse::SparseColMat<usize, f64>,
    ) -> Result<rerun::datatypes::TensorData, Box<dyn std::error::Error>> {
        // Fixed target size for visualization
        let target_size = 100;
        let target_rows = target_size;
        let target_cols = target_size;

        // Downsample to fixed size using block averaging
        let dense_matrix = Self::downsample_sparse_matrix(hessian, target_rows, target_cols);

        // Find min/max for normalization
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &val in &dense_matrix {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        let max_abs = max_val.abs().max(min_val.abs());

        // Convert to RGB bytes
        let mut rgb_data = Vec::with_capacity(target_rows * target_cols * 3);

        for &val in &dense_matrix {
            let rgb = Self::value_to_rgb_heatmap(val, max_abs);
            rgb_data.extend_from_slice(&rgb);
        }

        // Create tensor data
        let tensor = rerun::datatypes::TensorData::new(
            vec![target_rows as u64, target_cols as u64, 3],
            rerun::datatypes::TensorBuffer::U8(rgb_data.into()),
        );

        Ok(tensor)
    }

    /// Convert gradient vector to a fixed 100-width horizontal bar image.
    fn gradient_to_image(
        gradient: &[f64],
    ) -> Result<rerun::datatypes::TensorData, Box<dyn std::error::Error>> {
        let n = gradient.len();

        // Fixed size for visualization
        let bar_height = 50;
        let target_width = 100;

        // Find max absolute value for normalization
        let max_abs = gradient
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f64, |a, b| a.max(b));

        let mut rgb_data = Vec::with_capacity(bar_height * target_width * 3);

        for _ in 0..bar_height {
            for i in 0..target_width {
                // Average over a block of the gradient
                let start = (i * n) / target_width;
                let end = ((i + 1) * n) / target_width;
                let sum: f64 = gradient[start..end].iter().sum();
                let val = sum / (end - start).max(1) as f64;

                let rgb = Self::value_to_rgb_heatmap(val, max_abs);
                rgb_data.extend_from_slice(&rgb);
            }
        }

        let tensor = rerun::datatypes::TensorData::new(
            vec![bar_height as u64, target_width as u64, 3],
            rerun::datatypes::TensorBuffer::U8(rgb_data.into()),
        );

        Ok(tensor)
    }

    /// Downsample a sparse matrix to target size using block averaging.
    ///
    /// This function efficiently iterates only over non-zero elements using
    /// the sparse CSC (Compressed Sparse Column) format, avoiding O(m*n) complexity.
    fn downsample_sparse_matrix(
        sparse: &sparse::SparseColMat<usize, f64>,
        target_rows: usize,
        target_cols: usize,
    ) -> Vec<f64> {
        let m = sparse.nrows();
        let n = sparse.ncols();

        let mut downsampled = vec![0.0; target_rows * target_cols];
        let mut counts = vec![0usize; target_rows * target_cols];

        // Efficiently iterate only over non-zero entries using CSC format
        // This is O(nnz) instead of O(m*n), critical for large matrices!
        let symbolic = sparse.symbolic();

        for col in 0..n {
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = sparse.val_of_col(col);

            for (idx_in_col, &row) in row_indices.iter().enumerate() {
                let value = col_values[idx_in_col];

                if value.abs() > 1e-12 {
                    let target_row = (row * target_rows) / m;
                    let target_col = (col * target_cols) / n;
                    let idx = target_row * target_cols + target_col;

                    downsampled[idx] += value;
                    counts[idx] += 1;
                }
            }
        }

        // Average bins
        for i in 0..downsampled.len() {
            if counts[i] > 0 {
                downsampled[i] /= counts[i] as f64;
            }
        }

        downsampled
    }

    /// Map a scalar value to RGB color using white-to-blue gradient.
    ///
    /// Uses a simple white→blue gradient based on magnitude:
    /// - Low values: white (255, 255, 255)
    /// - High values: blue (0, 0, 255)
    ///
    /// # Arguments
    ///
    /// * `value` - The value to map (absolute value is used)
    /// * `max_abs` - Maximum absolute value for normalization
    ///
    /// # Returns
    ///
    /// RGB tuple as [r, g, b] bytes
    fn value_to_rgb_heatmap(value: f64, max_abs: f64) -> [u8; 3] {
        if !value.is_finite() || max_abs == 0.0 {
            return [255, 255, 255]; // White for invalid/zero
        }

        // Normalize to [0, 1] using absolute value
        let normalized = (value.abs() / max_abs).clamp(0.0, 1.0);

        if normalized < 1e-10 {
            // Near-zero: white
            [255, 255, 255]
        } else {
            // White (255,255,255) → Blue (0,0,255)
            // Keep blue channel at 255, reduce red and green
            let intensity = (normalized * 255.0) as u8;
            let remaining = 255 - intensity;
            [remaining, remaining, 255]
        }
    }
}

impl Default for OptimizationVisualizer {
    fn default() -> Self {
        Self::new(false).unwrap_or_else(|_| Self {
            rec: None,
            enabled: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualizer_creation() {
        let vis = OptimizationVisualizer::new(false);
        assert!(vis.is_ok());

        let vis = vis.unwrap();
        assert!(!vis.is_enabled());
    }

    #[test]
    fn test_rgb_heatmap_conversion() {
        // Test zero - should be white
        let rgb = OptimizationVisualizer::value_to_rgb_heatmap(0.0, 1.0);
        assert_eq!(rgb, [255, 255, 255]);

        // Test max positive - should be blue
        let rgb = OptimizationVisualizer::value_to_rgb_heatmap(1.0, 1.0);
        assert_eq!(rgb, [0, 0, 255]);

        // Test max negative (abs value) - should also be blue
        let rgb = OptimizationVisualizer::value_to_rgb_heatmap(-1.0, 1.0);
        assert_eq!(rgb, [0, 0, 255]);

        // Test mid-range value - should be light blue
        let rgb = OptimizationVisualizer::value_to_rgb_heatmap(0.5, 1.0);
        assert_eq!(rgb, [128, 128, 255]);
    }
}
