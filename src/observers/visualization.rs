//! Rerun observer for real-time optimization visualization.
//!
//! This module provides a Rerun-based observer that implements the `OptObserver` trait,
//! enabling clean separation between optimization logic and visualization.
//!
//! # Features
//!
//! - **Time series plots**: Cost, gradient norm, damping parameter, step quality
//! - **Sparse Hessian visualization**: Heat map showing matrix structure and values
//! - **Gradient visualization**: Vector representation with magnitude encoding
//! - **Manifold state**: Real-time pose updates for SE2/SE3 problems
//! - **Initial graph visualization**: Display starting configuration
//!
//! # Observer Pattern Integration
//!
//! Instead of being tightly coupled to the optimizer loop, `RerunObserver` implements
//! the `OptObserver` trait. Register it with any optimizer and it will automatically
//! receive updates at each iteration.
//!
//! # Feature Flag
//!
//! This module requires the `visualization` feature to be enabled:
//!
//! ```toml
//! apex-solver = { version = "0.1", features = ["visualization"] }
//! ```
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```no_run
//! use apex_solver::{LevenbergMarquardt, LevenbergMarquardtConfig};
//! use apex_solver::observers::RerunObserver;
//! # use apex_solver::core::problem::Problem;
//! # use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let problem = Problem::new();
//! # let initial_values = HashMap::new();
//!
//! let config = LevenbergMarquardtConfig::new().with_max_iterations(100);
//! let mut solver = LevenbergMarquardt::with_config(config);
//!
//! // Add Rerun visualization observer
//! let rerun_observer = RerunObserver::new(true)?;
//! solver.add_observer(rerun_observer);
//!
//! let result = solver.optimize(&problem, &initial_values)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Save to File Instead of Live Viewer
//!
//! ```no_run
//! # use apex_solver::observers::RerunObserver;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let rerun_observer = RerunObserver::new_with_options(
//!     true,
//!     Some("my_optimization.rrd")
//! )?;
//! // ... add to solver and optimize ...
//! # Ok(())
//! # }
//! ```

use crate::core::problem::VariableEnum;
use crate::io;
use crate::observers::OptObserver;
use faer::Mat;
use faer::sparse;
use std::cell::RefCell;
use std::collections::HashMap;
use tracing::{info, warn};

/// Rerun observer for real-time optimization visualization.
///
/// This observer logs comprehensive optimization data to Rerun for interactive
/// visualization and debugging. It implements the `OptObserver` trait, enabling
/// clean integration with any optimizer through the observer pattern.
///
/// # What Gets Visualized
///
/// - **Time series**: Cost, gradient norm, damping (LM), step norm, step quality
/// - **Matrices**: Sparse Hessian (downsampled heat map), gradient vector
/// - **Poses**: SE2/SE3 manifold states updated each iteration
/// - **Status**: Convergence information
///
/// # Observer Pattern Benefits
///
/// - Decoupled from optimizer internals
/// - Can be combined with other observers (CSV, metrics, etc.)
/// - No `#[cfg(feature = "visualization")]` scattered through optimizer code
/// - Easy to enable/disable without changing optimizer logic
///
/// # Performance
///
/// The observer is designed to have minimal overhead:
/// - Matrix visualizations use downsampling (100×100 for Hessian)
/// - Rerun logging is asynchronous
/// - When disabled, `is_enabled()` returns false immediately
pub struct RerunObserver {
    rec: Option<rerun::RecordingStream>,
    enabled: bool,
    // Mutable state for tracking optimizer-specific metrics
    // Using RefCell for interior mutability (observer receives &self)
    iteration_metrics: RefCell<IterationMetrics>,
}

/// Internal metrics tracked across iterations.
///
/// These are set by optimizer-specific methods (e.g., `set_iteration_metrics`)
/// and logged in the `on_step` callback.
#[derive(Default, Clone)]
struct IterationMetrics {
    cost: Option<f64>,
    gradient_norm: Option<f64>,
    damping: Option<f64>,
    step_norm: Option<f64>,
    step_quality: Option<f64>,
    hessian: Option<sparse::SparseColMat<usize, f64>>,
    gradient: Option<Mat<f64>>,
}

impl RerunObserver {
    /// Create a new Rerun observer.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable visualization
    ///
    /// # Returns
    ///
    /// A new observer instance that spawns a Rerun viewer (or saves to file if viewer unavailable).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::observers::RerunObserver;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let observer = RerunObserver::new(true)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(enabled: bool) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_options(enabled, None)
    }

    /// Create a new Rerun observer with file save option.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable visualization
    /// * `save_path` - Optional path to save recording to file instead of spawning viewer
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::observers::RerunObserver;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Save to file
    /// let observer = RerunObserver::new_with_options(true, Some("opt.rrd"))?;
    ///
    /// // Spawn live viewer
    /// let observer2 = RerunObserver::new_with_options(true, None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_options(
        enabled: bool,
        save_path: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let rec = if enabled {
            let rec = if let Some(path) = save_path {
                // Save to file
                info!("Saving visualization to: {}", path);
                rerun::RecordingStreamBuilder::new("apex-solver-optimization").save(path)?
            } else {
                // Try to spawn Rerun viewer
                match rerun::RecordingStreamBuilder::new("apex-solver-optimization").spawn() {
                    Ok(rec) => {
                        info!("Rerun viewer launched successfully");
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

        Ok(Self {
            rec,
            enabled,
            iteration_metrics: RefCell::new(IterationMetrics::default()),
        })
    }

    /// Check if visualization is enabled and active.
    #[inline(always)]
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.rec.is_some()
    }

    // ========================================================================
    // Public Methods for Optimizer-Specific Data
    // ========================================================================
    // These methods allow optimizers to provide additional context beyond
    // what's available in the OptObserver::on_step callback.
    // ========================================================================

    /// Set iteration metrics for the next on_step call.
    ///
    /// This method should be called by optimizers before notifying observers
    /// to provide context like cost, gradient norm, damping, etc.
    ///
    /// # Arguments
    ///
    /// * `cost` - Current cost value
    /// * `gradient_norm` - L2 norm of gradient
    /// * `damping` - Current damping parameter (LM-specific, use None for GN/DogLeg)
    /// * `step_norm` - L2 norm of parameter update
    /// * `step_quality` - Step quality metric ρ (actual vs predicted reduction)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use apex_solver::observers::RerunObserver;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let observer = RerunObserver::new(true)?;
    /// observer.set_iteration_metrics(
    ///     1.234,      // cost
    ///     0.056,      // gradient_norm
    ///     Some(0.01), // damping (LM only)
    ///     0.023,      // step_norm
    ///     Some(0.95), // step_quality
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_iteration_metrics(
        &self,
        cost: f64,
        gradient_norm: f64,
        damping: Option<f64>,
        step_norm: f64,
        step_quality: Option<f64>,
    ) {
        let mut metrics = self.iteration_metrics.borrow_mut();
        metrics.cost = Some(cost);
        metrics.gradient_norm = Some(gradient_norm);
        metrics.damping = damping;
        metrics.step_norm = Some(step_norm);
        metrics.step_quality = step_quality;
    }

    /// Set matrix data (Hessian and gradient) for visualization.
    ///
    /// This should be called before `on_step` if you want to visualize matrices.
    ///
    /// # Arguments
    ///
    /// * `hessian` - Optional sparse Hessian matrix (J^T J)
    /// * `gradient` - Optional gradient vector (J^T r)
    pub fn set_matrix_data(
        &self,
        hessian: Option<sparse::SparseColMat<usize, f64>>,
        gradient: Option<Mat<f64>>,
    ) {
        let mut metrics = self.iteration_metrics.borrow_mut();
        metrics.hessian = hessian;
        metrics.gradient = gradient;
    }

    /// Log the initial graph structure before optimization.
    ///
    /// This should be called once before optimization starts to visualize
    /// the initial configuration.
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

    /// Log convergence status and final summary.
    ///
    /// Call this after optimization completes.
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

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    /// Log scalar time series data.
    fn log_scalars(&self, iteration: usize, metrics: &IterationMetrics) {
        if !self.is_enabled() {
            return;
        }

        let rec = self.rec.as_ref().unwrap();
        rec.set_time_sequence("iteration", iteration as i64);

        // Log each metric to separate entity paths for independent scaling
        if let Some(cost) = metrics.cost {
            let _ = rec.log("cost_plot/value", &rerun::archetypes::Scalars::new([cost]));
        }

        if let Some(gradient_norm) = metrics.gradient_norm {
            let _ = rec.log(
                "gradient_plot/norm",
                &rerun::archetypes::Scalars::new([gradient_norm]),
            );
        }

        if let Some(damping) = metrics.damping {
            let _ = rec.log(
                "damping_plot/lambda",
                &rerun::archetypes::Scalars::new([damping]),
            );
        }

        if let Some(step_norm) = metrics.step_norm {
            let _ = rec.log(
                "step_plot/norm",
                &rerun::archetypes::Scalars::new([step_norm]),
            );
        }

        if let Some(step_quality) = metrics.step_quality {
            let _ = rec.log(
                "quality_plot/rho",
                &rerun::archetypes::Scalars::new([step_quality]),
            );
        }
    }

    /// Log matrix visualizations (Hessian and gradient).
    fn log_matrices(&self, iteration: usize, metrics: &IterationMetrics) {
        if !self.is_enabled() {
            return;
        }

        let rec = self.rec.as_ref().unwrap();
        rec.set_time_sequence("iteration", iteration as i64);

        // Log Hessian if available
        if let Some(ref hessian) = metrics.hessian
            && let Ok(image_data) = Self::sparse_hessian_to_image(hessian)
        {
            let _ = rec.log(
                "optimization/matrices/hessian",
                &rerun::archetypes::Tensor::new(image_data),
            );
        }

        // Log gradient if available
        if let Some(ref gradient) = metrics.gradient {
            let grad_vec: Vec<f64> = (0..gradient.nrows()).map(|i| gradient[(i, 0)]).collect();
            if let Ok(image_data) = Self::gradient_to_image(&grad_vec) {
                let _ = rec.log(
                    "optimization/matrices/gradient",
                    &rerun::archetypes::Tensor::new(image_data),
                );
            }
        }
    }

    /// Log manifold states (SE2/SE3 poses).
    fn log_manifolds(&self, iteration: usize, variables: &HashMap<String, VariableEnum>) {
        if !self.is_enabled() {
            return;
        }

        let rec = self.rec.as_ref().unwrap();
        rec.set_time_sequence("iteration", iteration as i64);

        for (var_name, var) in variables {
            match var {
                VariableEnum::SE3(v) => {
                    let trans = v.value.translation();
                    let rot = v.value.rotation_quaternion();

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

                    let _ = rec.log(
                        format!("optimized_graph/se3_poses/{}", var_name),
                        &transform,
                    );

                    let _ = rec.log(
                        format!("optimized_graph/se3_poses/{}", var_name),
                        &rerun::archetypes::Pinhole::from_fov_and_aspect_ratio(0.5, 1.0),
                    );
                }
                VariableEnum::SE2(v) => {
                    let x = v.value.x();
                    let y = v.value.y();

                    let position = rerun::external::glam::Vec3::new(x as f32, y as f32, 0.0);
                    let rotation = rerun::external::glam::Quat::IDENTITY;

                    let transform =
                        rerun::Transform3D::from_translation_rotation(position, rotation);

                    let _ = rec.log(
                        format!("optimized_graph/se2_poses/{}", var_name),
                        &transform,
                    );
                }
                _ => {
                    // Skip other manifold types (SO2, SO3, Rn)
                }
            }
        }
    }

    /// Convert sparse Hessian matrix to fixed 100×100 RGB image with heat map coloring.
    fn sparse_hessian_to_image(
        hessian: &sparse::SparseColMat<usize, f64>,
    ) -> Result<rerun::datatypes::TensorData, Box<dyn std::error::Error>> {
        let target_size = 100;
        let target_rows = target_size;
        let target_cols = target_size;

        let dense_matrix = Self::downsample_sparse_matrix(hessian, target_rows, target_cols);

        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &val in &dense_matrix {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        let max_abs = max_val.abs().max(min_val.abs());

        let mut rgb_data = Vec::with_capacity(target_rows * target_cols * 3);

        for &val in &dense_matrix {
            let rgb = Self::value_to_rgb_heatmap(val, max_abs);
            rgb_data.extend_from_slice(&rgb);
        }

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
        let bar_height = 50;
        let target_width = 100;

        let max_abs = gradient
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f64, |a, b| a.max(b));

        let mut rgb_data = Vec::with_capacity(bar_height * target_width * 3);

        for _ in 0..bar_height {
            for i in 0..target_width {
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
    fn downsample_sparse_matrix(
        sparse: &sparse::SparseColMat<usize, f64>,
        target_rows: usize,
        target_cols: usize,
    ) -> Vec<f64> {
        let m = sparse.nrows();
        let n = sparse.ncols();

        let mut downsampled = vec![0.0; target_rows * target_cols];
        let mut counts = vec![0usize; target_rows * target_cols];

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

        for i in 0..downsampled.len() {
            if counts[i] > 0 {
                downsampled[i] /= counts[i] as f64;
            }
        }

        downsampled
    }

    /// Map a scalar value to RGB color using white-to-blue gradient.
    fn value_to_rgb_heatmap(value: f64, max_abs: f64) -> [u8; 3] {
        if !value.is_finite() || max_abs == 0.0 {
            return [255, 255, 255];
        }

        let normalized = (value.abs() / max_abs).clamp(0.0, 1.0);

        if normalized < 1e-10 {
            [255, 255, 255]
        } else {
            let intensity = (normalized * 255.0) as u8;
            let remaining = 255 - intensity;
            [remaining, remaining, 255]
        }
    }
}

// ============================================================================
// OptObserver Trait Implementation
// ============================================================================

impl OptObserver for RerunObserver {
    /// Called at each optimization iteration.
    ///
    /// This logs all visualization data to Rerun, including:
    /// - Time series plots (cost, gradient, damping, step quality)
    /// - Matrix visualizations (Hessian, gradient) if set via `set_matrix_data`
    /// - Manifold states (SE2/SE3 poses)
    ///
    /// # Arguments
    ///
    /// * `values` - Current variable values (manifold states)
    /// * `iteration` - Current iteration number
    fn on_step(&self, values: &HashMap<String, VariableEnum>, iteration: usize) {
        if !self.is_enabled() {
            return;
        }

        let metrics = self.iteration_metrics.borrow();

        // Log all data types
        self.log_scalars(iteration, &metrics);
        self.log_matrices(iteration, &metrics);
        self.log_manifolds(iteration, values);

        // Clear transient data for next iteration
        drop(metrics);
        // Note: We don't clear the RefCell here to allow access from multiple threads
    }
}

impl Default for RerunObserver {
    fn default() -> Self {
        Self::new(false).unwrap_or_else(|_| Self {
            rec: None,
            enabled: false,
            iteration_metrics: RefCell::new(IterationMetrics::default()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observer_creation() {
        let observer = RerunObserver::new(false);
        assert!(observer.is_ok());

        let observer = observer.unwrap();
        assert!(!observer.is_enabled());
    }

    #[test]
    fn test_rgb_heatmap_conversion() {
        let rgb = RerunObserver::value_to_rgb_heatmap(0.0, 1.0);
        assert_eq!(rgb, [255, 255, 255]);

        let rgb = RerunObserver::value_to_rgb_heatmap(1.0, 1.0);
        assert_eq!(rgb, [0, 0, 255]);

        let rgb = RerunObserver::value_to_rgb_heatmap(-1.0, 1.0);
        assert_eq!(rgb, [0, 0, 255]);

        let rgb = RerunObserver::value_to_rgb_heatmap(0.5, 1.0);
        assert_eq!(rgb, [128, 128, 255]);
    }

    #[test]
    fn test_set_metrics() {
        let observer = RerunObserver::new(false).unwrap();
        observer.set_iteration_metrics(1.0, 0.5, Some(0.01), 0.1, Some(0.95));

        let metrics = observer.iteration_metrics.borrow();
        assert_eq!(metrics.cost, Some(1.0));
        assert_eq!(metrics.gradient_norm, Some(0.5));
        assert_eq!(metrics.damping, Some(0.01));
        assert_eq!(metrics.step_norm, Some(0.1));
        assert_eq!(metrics.step_quality, Some(0.95));
    }

    #[test]
    fn test_observer_trait() {
        let observer = RerunObserver::new(false).unwrap();
        let values = HashMap::new();

        // Should not panic when disabled
        observer.on_step(&values, 0);
        observer.on_step(&values, 1);
    }
}
