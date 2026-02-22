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
//! apex-solver = { version = "1.0", features = ["visualization"] }
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
use crate::observers::{ObserverError, ObserverResult, OptObserver};
use apex_io as io;
use apex_manifolds::se3::SE3;
use apex_manifolds::{LieGroup, ManifoldType};
use faer::Mat;
use faer::sparse;
use nalgebra::DVector;
use std::cell::RefCell;
use std::collections::HashMap;
use tracing::{info, warn};

// ============================================================================
// Visualization Mode
// ============================================================================

/// Controls when visualization updates occur during optimization.
///
/// This enum determines how frequently the observer logs visualization data
/// during the optimization process.
///
/// # Examples
///
/// ```no_run
/// use apex_solver::observers::{VisualizationConfig, VisualizationMode};
///
/// // Show every iteration (detailed but slower)
/// let config = VisualizationConfig::new()
///     .with_visualization_mode(VisualizationMode::Iterative);
///
/// // Show only initial and final (faster, default)
/// let config = VisualizationConfig::new()
///     .with_visualization_mode(VisualizationMode::InitialAndFinal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisualizationMode {
    /// Show updates at every optimization iteration (detailed but slower).
    /// Use this mode when you need to analyze the optimization trajectory.
    Iterative,

    /// Show only initial state and final state after optimization (default, faster).
    /// Use this mode for production runs or when only the result matters.
    #[default]
    InitialAndFinal,
}

// ============================================================================
// Visualization Configuration
// ============================================================================

/// Configuration for visualization options in RerunObserver.
///
/// This struct controls what elements are visualized and their appearance.
/// Use the builder pattern to customize settings.
///
/// # Examples
///
/// ```no_run
/// use apex_solver::observers::VisualizationConfig;
///
/// // Default: show everything
/// let config = VisualizationConfig::new();
///
/// // Cameras only with larger frustums
/// let config = VisualizationConfig::cameras_only()
///     .with_camera_fov(0.8)
///     .with_camera_frustum_scale(2.0);
///
/// // Landmarks only with larger points
/// let config = VisualizationConfig::landmarks_only()
///     .with_landmark_point_size(0.05);
///
/// // Bundle adjustment preset
/// let config = VisualizationConfig::for_bundle_adjustment();
/// ```
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    // === Element Visibility ===
    /// Show camera poses (SE3 variables) as frustums
    pub show_cameras: bool,
    /// Show 3D landmarks (Rn with dim=3) as point cloud
    pub show_landmarks: bool,
    /// Show SE2 poses (for 2D SLAM)
    pub show_se2_poses: bool,
    /// Show time series plots (cost, gradient, damping, etc.)
    pub show_plots: bool,
    /// Show matrix visualizations (Hessian, gradient)
    pub show_matrices: bool,

    // === Camera Frustum Settings ===
    /// Camera frustum field of view in radians (default: 0.5)
    pub camera_fov: f32,
    /// Camera frustum aspect ratio (default: 1.0)
    pub camera_aspect_ratio: f32,
    /// Scale factor for camera frustum size (default: 1.0)
    pub camera_frustum_scale: f32,

    // === Landmark Point Cloud Settings ===
    /// Radius of 3D landmark points (default: 0.02)
    pub landmark_point_size: f32,
    /// Color for initial landmarks RGB (default: blue [100, 150, 255])
    pub initial_landmark_color: [u8; 3],
    /// Color for optimized landmarks RGB (default: gold [255, 200, 50])
    pub optimized_landmark_color: [u8; 3],

    // === SE2 Pose Settings ===
    /// Radius for SE2 pose markers (default: 0.5)
    pub se2_pose_radius: f32,
    /// Color for initial SE2 poses RGB (default: blue [100, 150, 255])
    pub initial_se2_color: [u8; 3],

    // === Matrix Visualization Settings ===
    /// Target size for Hessian downsampling (default: 100)
    pub hessian_downsample_size: usize,
    /// Target width for gradient visualization (default: 100)
    pub gradient_bar_width: usize,

    // === General Settings ===
    /// Graph scale factor for pose graph visualization
    pub graph_scale: f32,
    /// Invert camera poses for display (T_wc -> T_cw for BA)
    pub invert_camera_poses: bool,
    /// Visualization mode: iterative or initial-and-final only
    pub visualization_mode: VisualizationMode,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            // Element visibility - all enabled by default
            show_cameras: true,
            show_landmarks: true,
            show_se2_poses: true,
            show_plots: true,
            show_matrices: true,

            // Camera frustum settings (matching previous hardcoded values)
            camera_fov: 0.5,
            camera_aspect_ratio: 1.0,
            camera_frustum_scale: 1.0,

            // Landmark settings (matching previous hardcoded values)
            landmark_point_size: 0.02,
            initial_landmark_color: [100, 150, 255],  // Blue
            optimized_landmark_color: [255, 200, 50], // Gold

            // SE2 pose settings
            se2_pose_radius: 0.5,
            initial_se2_color: [100, 150, 255], // Blue

            // Matrix visualization
            hessian_downsample_size: 100,
            gradient_bar_width: 100,

            // General
            graph_scale: 1.0,
            invert_camera_poses: false,
            visualization_mode: VisualizationMode::default(),
        }
    }
}

impl VisualizationConfig {
    /// Create a new configuration with default values (show everything).
    pub fn new() -> Self {
        Self::default()
    }

    // === Element Visibility Builders ===

    /// Set whether to show camera poses (SE3 frustums).
    pub fn with_show_cameras(mut self, show: bool) -> Self {
        self.show_cameras = show;
        self
    }

    /// Set whether to show 3D landmarks.
    pub fn with_show_landmarks(mut self, show: bool) -> Self {
        self.show_landmarks = show;
        self
    }

    /// Set whether to show SE2 poses.
    pub fn with_show_se2_poses(mut self, show: bool) -> Self {
        self.show_se2_poses = show;
        self
    }

    /// Set whether to show time series plots.
    pub fn with_show_plots(mut self, show: bool) -> Self {
        self.show_plots = show;
        self
    }

    /// Set whether to show matrix visualizations.
    pub fn with_show_matrices(mut self, show: bool) -> Self {
        self.show_matrices = show;
        self
    }

    // === Camera Settings Builders ===

    /// Set camera frustum field of view in radians.
    pub fn with_camera_fov(mut self, fov: f32) -> Self {
        self.camera_fov = fov;
        self
    }

    /// Set camera frustum aspect ratio.
    pub fn with_camera_aspect_ratio(mut self, ratio: f32) -> Self {
        self.camera_aspect_ratio = ratio;
        self
    }

    /// Set camera frustum scale factor.
    pub fn with_camera_frustum_scale(mut self, scale: f32) -> Self {
        self.camera_frustum_scale = scale;
        self
    }

    // === Landmark Settings Builders ===

    /// Set 3D landmark point size/radius.
    pub fn with_landmark_point_size(mut self, size: f32) -> Self {
        self.landmark_point_size = size;
        self
    }

    /// Set color for initial landmarks (RGB).
    pub fn with_initial_landmark_color(mut self, rgb: [u8; 3]) -> Self {
        self.initial_landmark_color = rgb;
        self
    }

    /// Set color for optimized landmarks (RGB).
    pub fn with_optimized_landmark_color(mut self, rgb: [u8; 3]) -> Self {
        self.optimized_landmark_color = rgb;
        self
    }

    // === SE2 Settings Builders ===

    /// Set SE2 pose marker radius.
    pub fn with_se2_pose_radius(mut self, radius: f32) -> Self {
        self.se2_pose_radius = radius;
        self
    }

    /// Set color for initial SE2 poses (RGB).
    pub fn with_initial_se2_color(mut self, rgb: [u8; 3]) -> Self {
        self.initial_se2_color = rgb;
        self
    }

    // === Matrix Settings Builders ===

    /// Set target size for Hessian downsampling.
    pub fn with_hessian_downsample_size(mut self, size: usize) -> Self {
        self.hessian_downsample_size = size;
        self
    }

    /// Set target width for gradient visualization.
    pub fn with_gradient_bar_width(mut self, width: usize) -> Self {
        self.gradient_bar_width = width;
        self
    }

    // === General Settings Builders ===

    /// Set graph scale factor.
    pub fn with_graph_scale(mut self, scale: f32) -> Self {
        self.graph_scale = scale;
        self
    }

    /// Set whether to invert camera poses (T_wc -> T_cw).
    pub fn with_invert_camera_poses(mut self, invert: bool) -> Self {
        self.invert_camera_poses = invert;
        self
    }

    /// Set visualization mode (iterative or initial-and-final).
    ///
    /// # Arguments
    ///
    /// * `mode` - The visualization mode to use
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::observers::{VisualizationConfig, VisualizationMode};
    ///
    /// let config = VisualizationConfig::new()
    ///     .with_visualization_mode(VisualizationMode::Iterative);
    /// ```
    pub fn with_visualization_mode(mut self, mode: VisualizationMode) -> Self {
        self.visualization_mode = mode;
        self
    }

    // === Convenience Presets ===

    /// Create a configuration that shows only camera poses (no landmarks).
    ///
    /// Useful for pose graph visualization or when landmarks are too dense.
    pub fn cameras_only() -> Self {
        Self::default()
            .with_show_cameras(true)
            .with_show_landmarks(false)
            .with_show_se2_poses(false)
    }

    /// Create a configuration that shows only 3D landmarks (no cameras).
    ///
    /// Useful for structure-from-motion point cloud visualization.
    pub fn landmarks_only() -> Self {
        Self::default()
            .with_show_cameras(false)
            .with_show_landmarks(true)
            .with_show_se2_poses(false)
    }

    /// Create a configuration optimized for bundle adjustment.
    ///
    /// - Inverts camera poses (T_wc -> T_cw for correct display)
    /// - Disables SE2 poses (not used in BA)
    pub fn for_bundle_adjustment() -> Self {
        Self::default()
            .with_invert_camera_poses(true)
            .with_show_se2_poses(false)
    }

    /// Create a configuration optimized for pose graph optimization.
    ///
    /// - No camera pose inversion (poses are already T_cw)
    /// - Disables landmarks (pose graphs don't have 3D points)
    pub fn for_pose_graph() -> Self {
        Self::default()
            .with_show_landmarks(false)
            .with_invert_camera_poses(false)
    }
}

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
/// - **3D Landmarks**: Rn variables with dimension=3 visualized as point clouds
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
/// - 3D landmarks are batch-logged as a single point cloud for efficiency
///
/// # Pose Convention Support
///
/// For bundle adjustment (BAL datasets), camera poses are stored as world-to-camera
/// transforms (T_wc). Set `invert_camera_poses = true` to display cameras correctly
/// by converting to camera-to-world (T_cw) convention for Rerun visualization.
pub struct RerunObserver {
    rec: Option<rerun::RecordingStream>,
    enabled: bool,
    // Mutable state for tracking optimizer-specific metrics
    // Using RefCell for interior mutability (observer receives &self)
    iteration_metrics: RefCell<IterationMetrics>,
    // Visualization configuration
    config: VisualizationConfig,
    // Cached initial positions for displacement visualization
    initial_camera_positions: RefCell<HashMap<String, [f32; 3]>>,
    initial_landmark_positions: RefCell<HashMap<String, [f32; 3]>>,
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
    pub fn new(enabled: bool) -> ObserverResult<Self> {
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
    pub fn new_with_options(enabled: bool, save_path: Option<&str>) -> ObserverResult<Self> {
        Self::with_config(enabled, save_path, VisualizationConfig::default())
    }

    /// Create a new Rerun observer with full configuration.
    ///
    /// This is the primary constructor for full control over visualization.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable visualization
    /// * `save_path` - Optional path to save recording to file instead of spawning viewer
    /// * `config` - Visualization configuration
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::observers::{RerunObserver, VisualizationConfig};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = VisualizationConfig::new()
    ///     .with_show_cameras(true)
    ///     .with_show_landmarks(false)
    ///     .with_camera_fov(0.8);
    ///
    /// let observer = RerunObserver::with_config(true, None, config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config(
        enabled: bool,
        save_path: Option<&str>,
        config: VisualizationConfig,
    ) -> ObserverResult<Self> {
        let rec = if enabled {
            let rec = if let Some(path) = save_path {
                // Save to file
                info!("Saving visualization to: {}", path);
                rerun::RecordingStreamBuilder::new("apex-solver-optimization")
                    .save(path)
                    .map_err(|e| {
                        ObserverError::RecordingSaveFailed {
                            path: path.to_string(),
                            reason: format!("{}", e),
                        }
                        .log_with_source(e)
                    })?
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
                            .save("optimization.rrd")
                            .map_err(|e2| {
                                ObserverError::RecordingSaveFailed {
                                    path: "optimization.rrd".to_string(),
                                    reason: format!("{}", e2),
                                }
                                .log_with_source(e2)
                            })?
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
            config,
            initial_camera_positions: RefCell::new(HashMap::new()),
            initial_landmark_positions: RefCell::new(HashMap::new()),
        })
    }

    /// Create a new Rerun observer configured for bundle adjustment.
    ///
    /// This constructor is designed for bundle adjustment / structure-from-motion
    /// problems where camera poses are stored in world-to-camera convention (T_wc)
    /// but need to be displayed in camera-to-world convention (T_cw).
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable visualization
    /// * `save_path` - Optional path to save recording to file instead of spawning viewer
    /// * `invert_camera_poses` - If true, invert SE3 poses before logging (T_wc -> T_cw)
    ///
    /// # Use Cases
    ///
    /// - **Pose graph optimization**: Use `invert_camera_poses = false` (poses are already T_cw)
    /// - **Bundle adjustment (BAL)**: Use `invert_camera_poses = true` (BAL stores T_wc)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::observers::RerunObserver;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // For bundle adjustment with BAL datasets (world-to-camera poses)
    /// let observer = RerunObserver::new_for_bundle_adjustment(true, None, true)?;
    ///
    /// // For pose graph optimization (camera-to-world poses)
    /// let observer = RerunObserver::new_for_bundle_adjustment(true, None, false)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_for_bundle_adjustment(
        enabled: bool,
        save_path: Option<&str>,
        invert_camera_poses: bool,
    ) -> ObserverResult<Self> {
        let config = VisualizationConfig::for_bundle_adjustment()
            .with_invert_camera_poses(invert_camera_poses);
        Self::with_config(enabled, save_path, config)
    }

    /// Get the current visualization configuration.
    pub fn config(&self) -> &VisualizationConfig {
        &self.config
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
    pub fn log_initial_graph(&self, graph: &io::Graph, scale: f32) -> ObserverResult<()> {
        let rec = self.rec.as_ref().ok_or_else(|| {
            ObserverError::InvalidState("Recording stream not initialized".to_string())
        })?;

        // Visualize SE3 vertices only (no edges)
        if self.config.show_cameras {
            for (id, vertex) in &graph.vertices_se3 {
                let (position, rotation) = vertex.to_rerun_transform(scale);
                let transform = rerun::Transform3D::from_translation_rotation(position, rotation);

                let entity_path = format!("initial_graph/se3_poses/{}", id);
                rec.log(entity_path.as_str(), &transform).map_err(|e| {
                    ObserverError::LoggingFailed {
                        entity_path: entity_path.clone(),
                        reason: format!("{}", e),
                    }
                    .log_with_source(e)
                })?;

                // Add a small pinhole camera for better visualization
                rec.log(
                    entity_path.as_str(),
                    &rerun::archetypes::Pinhole::from_fov_and_aspect_ratio(
                        self.config.camera_fov,
                        self.config.camera_aspect_ratio,
                    ),
                )
                .map_err(|e| {
                    ObserverError::LoggingFailed {
                        entity_path: entity_path.clone(),
                        reason: format!("{}", e),
                    }
                    .log_with_source(e)
                })?;
            }
        }

        // Visualize SE2 vertices only (no edges)
        if self.config.show_se2_poses && !graph.vertices_se2.is_empty() {
            let positions: Vec<[f32; 2]> = graph
                .vertices_se2
                .values()
                .map(|vertex| vertex.to_rerun_position_2d(scale))
                .collect();

            let color = self.config.initial_se2_color;
            let colors = vec![
                rerun::components::Color::from_rgb(color[0], color[1], color[2]);
                positions.len()
            ];

            rec.log(
                "initial_graph/se2_poses",
                &rerun::archetypes::Points2D::new(positions)
                    .with_colors(colors)
                    .with_radii([self.config.se2_pose_radius * scale]),
            )
            .map_err(|e| {
                ObserverError::LoggingFailed {
                    entity_path: "initial_graph/se2_poses".to_string(),
                    reason: format!("{}", e),
                }
                .log_with_source(e)
            })?;
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
    pub fn log_convergence(&self, status: &str) -> ObserverResult<()> {
        let rec = self.rec.as_ref().ok_or_else(|| {
            ObserverError::InvalidState("Recording stream not initialized".to_string())
        })?;

        // Log as a text annotation
        rec.log(
            "optimization/status",
            &rerun::archetypes::TextDocument::new(status),
        )
        .map_err(|e| {
            ObserverError::LoggingFailed {
                entity_path: "optimization/status".to_string(),
                reason: format!("{}", e),
            }
            .log_with_source(e)
        })?;

        Ok(())
    }

    /// Log initial bundle adjustment state before optimization.
    ///
    /// This method visualizes the initial camera poses and 3D landmarks
    /// before optimization begins, allowing comparison with optimized results.
    ///
    /// # Arguments
    ///
    /// * `initial_values` - Initial variable values from problem setup
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::observers::RerunObserver;
    /// use apex_solver::manifold::ManifoldType;
    /// use nalgebra::DVector;
    /// use std::collections::HashMap;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let observer = RerunObserver::new_for_bundle_adjustment(true, None, true)?;
    ///
    /// let mut initial_values = HashMap::new();
    /// // ... populate with camera poses and landmarks ...
    ///
    /// observer.log_initial_ba_state(&initial_values)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn log_initial_ba_state(
        &self,
        initial_values: &HashMap<String, (ManifoldType, DVector<f64>)>,
    ) -> ObserverResult<()> {
        let rec = self.rec.as_ref().ok_or_else(|| {
            ObserverError::InvalidState("Recording stream not initialized".to_string())
        })?;

        // Collect 3D landmarks for batch logging
        let mut landmark_positions: Vec<[f32; 3]> = Vec::new();
        let mut landmark_names: Vec<String> = Vec::new();

        // Get mutable access to caches
        let mut camera_cache = self.initial_camera_positions.borrow_mut();
        let mut landmark_cache = self.initial_landmark_positions.borrow_mut();

        for (var_name, (manifold_type, data)) in initial_values {
            match manifold_type {
                ManifoldType::SE3 if self.config.show_cameras => {
                    // Parse SE3 from data vector
                    let se3 = SE3::from(data.clone());

                    // Apply pose inversion if configured (for BA: T_wc -> T_cw)
                    let pose = if self.config.invert_camera_poses {
                        se3.inverse(None)
                    } else {
                        se3
                    };

                    let trans = pose.translation();
                    let rot = pose.rotation_quaternion();

                    // Cache the initial camera position for displacement calculation
                    camera_cache.insert(
                        var_name.clone(),
                        [trans.x as f32, trans.y as f32, trans.z as f32],
                    );

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

                    let entity_path = format!("initial_graph/cameras/{}", var_name);
                    rec.log(entity_path.as_str(), &transform).map_err(|e| {
                        ObserverError::LoggingFailed {
                            entity_path: entity_path.clone(),
                            reason: format!("{}", e),
                        }
                        .log_with_source(e)
                    })?;

                    rec.log(
                        entity_path.as_str(),
                        &rerun::archetypes::Pinhole::from_fov_and_aspect_ratio(
                            self.config.camera_fov,
                            self.config.camera_aspect_ratio,
                        ),
                    )
                    .map_err(|e| {
                        ObserverError::LoggingFailed {
                            entity_path: entity_path.clone(),
                            reason: format!("{}", e),
                        }
                        .log_with_source(e)
                    })?;
                }
                ManifoldType::RN if self.config.show_landmarks => {
                    // Handle 3D landmarks (Rn with dimension 3)
                    if data.len() == 3 {
                        let pos = [data[0] as f32, data[1] as f32, data[2] as f32];
                        landmark_positions.push(pos);
                        landmark_names.push(var_name.clone());

                        // Cache the initial landmark position for displacement calculation
                        landmark_cache.insert(var_name.clone(), pos);
                    }
                    // Skip non-3D Rn variables (e.g., camera intrinsics)
                }
                _ => {
                    // Skip other manifold types (SE2, SO2, SO3) or disabled types
                }
            }
        }

        // Batch log all initial landmarks as a single point cloud
        if self.config.show_landmarks && !landmark_positions.is_empty() {
            let color = self.config.initial_landmark_color;
            rec.log(
                "initial_graph/landmarks",
                &rerun::archetypes::Points3D::new(landmark_positions)
                    .with_radii([self.config.landmark_point_size])
                    .with_colors([rerun::components::Color::from_rgb(
                        color[0], color[1], color[2],
                    )]),
            )
            .map_err(|e| {
                ObserverError::LoggingFailed {
                    entity_path: "initial_graph/landmarks".to_string(),
                    reason: format!("{}", e),
                }
                .log_with_source(e)
            })?;
        }

        info!(
            "Logged initial BA state: {} cameras, {} landmarks",
            camera_cache.len(),
            landmark_cache.len()
        );

        Ok(())
    }

    /// Log the final optimized state after optimization completes.
    ///
    /// This method visualizes the final camera poses and 3D landmarks
    /// in a separate entity group ("final_graph/") to allow comparison
    /// with the initial state.
    ///
    /// # Arguments
    ///
    /// * `values` - Final optimized variable values
    /// * `iterations` - Total number of iterations performed
    fn log_final_state(
        &self,
        values: &HashMap<String, VariableEnum>,
        iterations: usize,
    ) -> ObserverResult<()> {
        let rec = self.rec.as_ref().ok_or_else(|| {
            ObserverError::InvalidState("Recording stream not initialized".to_string())
        })?;

        // Set time to final iteration
        rec.set_time_sequence("iteration", iterations as i64);

        // Collect final 3D landmarks for batch logging
        let mut final_landmark_positions: Vec<[f32; 3]> = Vec::new();
        let mut camera_count = 0;

        for (var_name, var) in values {
            match var {
                VariableEnum::SE3(v) if self.config.show_cameras => {
                    // Apply pose inversion if configured (for BA: T_wc -> T_cw)
                    let pose = if self.config.invert_camera_poses {
                        v.value.inverse(None)
                    } else {
                        v.value.clone()
                    };

                    let trans = pose.translation();
                    let rot = pose.rotation_quaternion();

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

                    let entity_path = format!("final_graph/cameras/{}", var_name);
                    rec.log(entity_path.as_str(), &transform).map_err(|e| {
                        ObserverError::LoggingFailed {
                            entity_path: entity_path.clone(),
                            reason: format!("{}", e),
                        }
                        .log_with_source(e)
                    })?;

                    rec.log(
                        entity_path.as_str(),
                        &rerun::archetypes::Pinhole::from_fov_and_aspect_ratio(
                            self.config.camera_fov,
                            self.config.camera_aspect_ratio,
                        ),
                    )
                    .map_err(|e| {
                        ObserverError::LoggingFailed {
                            entity_path: entity_path.clone(),
                            reason: format!("{}", e),
                        }
                        .log_with_source(e)
                    })?;

                    camera_count += 1;
                }
                VariableEnum::Rn(v) if self.config.show_landmarks => {
                    // Handle 3D landmarks (Rn with dimension 3)
                    let data = v.value.data();
                    if data.len() == 3 {
                        final_landmark_positions.push([
                            data[0] as f32,
                            data[1] as f32,
                            data[2] as f32,
                        ]);
                    }
                }
                _ => {
                    // Skip other manifold types
                }
            }
        }

        // Batch log all final landmarks as a single point cloud with green color
        if self.config.show_landmarks && !final_landmark_positions.is_empty() {
            // Use green for final/optimized landmarks to distinguish from initial (blue)
            let final_color: [u8; 3] = [50, 200, 100]; // Green
            rec.log(
                "final_graph/landmarks",
                &rerun::archetypes::Points3D::new(final_landmark_positions.clone())
                    .with_radii([self.config.landmark_point_size])
                    .with_colors([rerun::components::Color::from_rgb(
                        final_color[0],
                        final_color[1],
                        final_color[2],
                    )]),
            )
            .map_err(|e| {
                ObserverError::LoggingFailed {
                    entity_path: "final_graph/landmarks".to_string(),
                    reason: format!("{}", e),
                }
                .log_with_source(e)
            })?;
        }

        info!(
            "Logged final BA state after {} iterations: {} cameras, {} landmarks",
            iterations,
            camera_count,
            final_landmark_positions.len()
        );

        // Log displacement statistics
        self.log_displacement_statistics(values)?;

        Ok(())
    }

    /// Log displacement statistics comparing initial and final states.
    ///
    /// Calculates and logs the displacement of cameras and landmarks
    /// from their initial positions to their final optimized positions.
    fn log_displacement_statistics(
        &self,
        values: &HashMap<String, VariableEnum>,
    ) -> ObserverResult<()> {
        let initial_cameras = self.initial_camera_positions.borrow();
        let initial_landmarks = self.initial_landmark_positions.borrow();

        // Calculate camera displacements
        let mut camera_displacements: Vec<f32> = Vec::new();
        for (name, var) in values {
            if let VariableEnum::SE3(v) = var {
                // Apply same pose inversion as in initial state
                let pose = if self.config.invert_camera_poses {
                    v.value.inverse(None)
                } else {
                    v.value.clone()
                };

                if let Some(initial_pos) = initial_cameras.get(name) {
                    let final_pos = pose.translation();
                    let dx = final_pos.x as f32 - initial_pos[0];
                    let dy = final_pos.y as f32 - initial_pos[1];
                    let dz = final_pos.z as f32 - initial_pos[2];
                    let displacement = (dx * dx + dy * dy + dz * dz).sqrt();
                    camera_displacements.push(displacement);
                }
            }
        }

        // Calculate landmark displacements
        let mut landmark_displacements: Vec<f32> = Vec::new();
        for (name, var) in values {
            if let VariableEnum::Rn(v) = var {
                let data = v.value.data();
                if data.len() == 3
                    && let Some(initial_pos) = initial_landmarks.get(name)
                {
                    let dx = data[0] as f32 - initial_pos[0];
                    let dy = data[1] as f32 - initial_pos[1];
                    let dz = data[2] as f32 - initial_pos[2];
                    let displacement = (dx * dx + dy * dy + dz * dz).sqrt();
                    landmark_displacements.push(displacement);
                }
            }
        }

        // Log camera displacement statistics
        if !camera_displacements.is_empty() {
            let avg = camera_displacements.iter().sum::<f32>() / camera_displacements.len() as f32;
            let max = camera_displacements.iter().cloned().fold(0.0f32, f32::max);
            let min = camera_displacements
                .iter()
                .cloned()
                .fold(f32::MAX, f32::min);
            info!(
                "Camera displacement: avg={:.6}, min={:.6}, max={:.6} ({} cameras)",
                avg,
                min,
                max,
                camera_displacements.len()
            );
        }

        // Log landmark displacement statistics
        if !landmark_displacements.is_empty() {
            let avg =
                landmark_displacements.iter().sum::<f32>() / landmark_displacements.len() as f32;
            let max = landmark_displacements
                .iter()
                .cloned()
                .fold(0.0f32, f32::max);
            let min = landmark_displacements
                .iter()
                .cloned()
                .fold(f32::MAX, f32::min);
            info!(
                "Landmark displacement: avg={:.6}, min={:.6}, max={:.6} ({} landmarks)",
                avg,
                min,
                max,
                landmark_displacements.len()
            );
        }

        Ok(())
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    /// Log scalar time series data.
    fn log_scalars(&self, iteration: usize, metrics: &IterationMetrics) -> ObserverResult<()> {
        if !self.config.show_plots {
            return Ok(());
        }

        let rec = self.rec.as_ref().ok_or_else(|| {
            ObserverError::InvalidState("Recording stream not initialized".to_string())
        })?;
        rec.set_time_sequence("iteration", iteration as i64);

        // Log each metric to separate entity paths for independent scaling
        if let Some(cost) = metrics.cost {
            rec.log("cost_plot/value", &rerun::archetypes::Scalars::new([cost]))
                .map_err(|e| {
                    ObserverError::LoggingFailed {
                        entity_path: "cost_plot/value".to_string(),
                        reason: format!("{}", e),
                    }
                    .log_with_source(e)
                })?;
        }

        if let Some(gradient_norm) = metrics.gradient_norm {
            rec.log(
                "gradient_plot/norm",
                &rerun::archetypes::Scalars::new([gradient_norm]),
            )
            .map_err(|e| {
                ObserverError::LoggingFailed {
                    entity_path: "gradient_plot/norm".to_string(),
                    reason: format!("{}", e),
                }
                .log_with_source(e)
            })?;
        }

        if let Some(damping) = metrics.damping {
            rec.log(
                "damping_plot/lambda",
                &rerun::archetypes::Scalars::new([damping]),
            )
            .map_err(|e| {
                ObserverError::LoggingFailed {
                    entity_path: "damping_plot/lambda".to_string(),
                    reason: format!("{}", e),
                }
                .log_with_source(e)
            })?;
        }

        if let Some(step_norm) = metrics.step_norm {
            rec.log(
                "step_plot/norm",
                &rerun::archetypes::Scalars::new([step_norm]),
            )
            .map_err(|e| {
                ObserverError::LoggingFailed {
                    entity_path: "step_plot/norm".to_string(),
                    reason: format!("{}", e),
                }
                .log_with_source(e)
            })?;
        }

        if let Some(step_quality) = metrics.step_quality {
            rec.log(
                "quality_plot/rho",
                &rerun::archetypes::Scalars::new([step_quality]),
            )
            .map_err(|e| {
                ObserverError::LoggingFailed {
                    entity_path: "quality_plot/rho".to_string(),
                    reason: format!("{}", e),
                }
                .log_with_source(e)
            })?;
        }

        Ok(())
    }

    /// Log matrix visualizations (Hessian and gradient).
    fn log_matrices(&self, iteration: usize, metrics: &IterationMetrics) -> ObserverResult<()> {
        if !self.config.show_matrices {
            return Ok(());
        }

        let rec = self.rec.as_ref().ok_or_else(|| {
            ObserverError::InvalidState("Recording stream not initialized".to_string())
        })?;
        rec.set_time_sequence("iteration", iteration as i64);

        // Log Hessian if available
        if let Some(ref hessian) = metrics.hessian
            && let Ok(image_data) = self.sparse_hessian_to_image(hessian)
        {
            rec.log(
                "optimization/matrices/hessian",
                &rerun::archetypes::Tensor::new(image_data),
            )
            .map_err(|e| {
                ObserverError::LoggingFailed {
                    entity_path: "optimization/matrices/hessian".to_string(),
                    reason: format!("{}", e),
                }
                .log_with_source(e)
            })?;
        }

        // Log gradient if available
        if let Some(ref gradient) = metrics.gradient {
            let grad_vec: Vec<f64> = (0..gradient.nrows()).map(|i| gradient[(i, 0)]).collect();
            if let Ok(image_data) = self.gradient_to_image(&grad_vec) {
                rec.log(
                    "optimization/matrices/gradient",
                    &rerun::archetypes::Tensor::new(image_data),
                )
                .map_err(|e| {
                    ObserverError::LoggingFailed {
                        entity_path: "optimization/matrices/gradient".to_string(),
                        reason: format!("{}", e),
                    }
                    .log_with_source(e)
                })?;
            }
        }

        Ok(())
    }

    /// Log manifold states (SE2/SE3 poses and Rn 3D landmarks).
    fn log_manifolds(
        &self,
        iteration: usize,
        variables: &HashMap<String, VariableEnum>,
    ) -> ObserverResult<()> {
        let rec = self.rec.as_ref().ok_or_else(|| {
            ObserverError::InvalidState("Recording stream not initialized".to_string())
        })?;
        rec.set_time_sequence("iteration", iteration as i64);

        // Collect 3D landmarks for batch logging (much more efficient for large point clouds)
        let mut landmark_positions: Vec<[f32; 3]> = Vec::new();

        for (var_name, var) in variables {
            match var {
                VariableEnum::SE3(v) if self.config.show_cameras => {
                    // Apply pose inversion if configured (for BA: T_wc -> T_cw)
                    let pose = if self.config.invert_camera_poses {
                        v.value.inverse(None)
                    } else {
                        v.value.clone()
                    };

                    let trans = pose.translation();
                    let rot = pose.rotation_quaternion();

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

                    let entity_path = format!("optimized_graph/cameras/{}", var_name);
                    rec.log(entity_path.as_str(), &transform).map_err(|e| {
                        ObserverError::LoggingFailed {
                            entity_path: entity_path.clone(),
                            reason: format!("{}", e),
                        }
                        .log_with_source(e)
                    })?;

                    rec.log(
                        entity_path.as_str(),
                        &rerun::archetypes::Pinhole::from_fov_and_aspect_ratio(
                            self.config.camera_fov,
                            self.config.camera_aspect_ratio,
                        ),
                    )
                    .map_err(|e| {
                        ObserverError::LoggingFailed {
                            entity_path: entity_path.clone(),
                            reason: format!("{}", e),
                        }
                        .log_with_source(e)
                    })?;
                }
                VariableEnum::SE2(v) if self.config.show_se2_poses => {
                    let x = v.value.x();
                    let y = v.value.y();

                    let position = rerun::external::glam::Vec3::new(x as f32, y as f32, 0.0);
                    let rotation = rerun::external::glam::Quat::IDENTITY;

                    let transform =
                        rerun::Transform3D::from_translation_rotation(position, rotation);

                    let entity_path = format!("optimized_graph/se2_poses/{}", var_name);
                    rec.log(entity_path.as_str(), &transform).map_err(|e| {
                        ObserverError::LoggingFailed {
                            entity_path: entity_path.clone(),
                            reason: format!("{}", e),
                        }
                        .log_with_source(e)
                    })?;
                }
                VariableEnum::Rn(v) if self.config.show_landmarks => {
                    // Handle 3D landmarks (Rn with dimension 3)
                    let data = v.value.data();
                    if data.len() == 3 {
                        landmark_positions.push([data[0] as f32, data[1] as f32, data[2] as f32]);
                    }
                    // Skip non-3D Rn variables (e.g., camera intrinsics)
                }
                _ => {
                    // Skip other manifold types (SO2, SO3) or disabled types
                }
            }
        }

        // Batch log all 3D landmarks as a single point cloud (efficient for 100K+ points)
        if self.config.show_landmarks && !landmark_positions.is_empty() {
            let color = self.config.optimized_landmark_color;
            rec.log(
                "optimized_graph/landmarks",
                &rerun::archetypes::Points3D::new(landmark_positions)
                    .with_radii([self.config.landmark_point_size])
                    .with_colors([rerun::components::Color::from_rgb(
                        color[0], color[1], color[2],
                    )]),
            )
            .map_err(|e| {
                ObserverError::LoggingFailed {
                    entity_path: "optimized_graph/landmarks".to_string(),
                    reason: format!("{}", e),
                }
                .log_with_source(e)
            })?;
        }

        Ok(())
    }

    /// Convert sparse Hessian matrix to RGB image with heat map coloring.
    fn sparse_hessian_to_image(
        &self,
        hessian: &sparse::SparseColMat<usize, f64>,
    ) -> ObserverResult<rerun::datatypes::TensorData> {
        let target_size = self.config.hessian_downsample_size;
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

    /// Convert gradient vector to a horizontal bar image.
    fn gradient_to_image(&self, gradient: &[f64]) -> ObserverResult<rerun::datatypes::TensorData> {
        let n = gradient.len();
        let bar_height = 50;
        let target_width = self.config.gradient_bar_width;

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
    /// In `InitialAndFinal` mode, this method only logs scalar metrics (plots)
    /// during intermediate iterations. The full manifold state is logged at
    /// iteration 0 (initial) and in `on_optimization_complete` (final).
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

        // Always log scalar metrics (plots) - they're lightweight and useful
        if let Err(e) = self.log_scalars(iteration, &metrics) {
            let _ = e.log();
        }

        // In InitialAndFinal mode, skip manifold logging for intermediate iterations
        // Initial state (iteration 0) is logged via log_initial_ba_state()
        // Final state is logged via on_optimization_complete()
        let should_log_manifolds = match self.config.visualization_mode {
            VisualizationMode::Iterative => true,
            VisualizationMode::InitialAndFinal => false, // Skip intermediate iterations
        };

        if should_log_manifolds {
            if let Err(e) = self.log_matrices(iteration, &metrics) {
                let _ = e.log();
            }
            if let Err(e) = self.log_manifolds(iteration, values) {
                let _ = e.log();
            }
        }

        // Clear transient data for next iteration
        drop(metrics);
        // Note: We don't clear the RefCell here to allow access from multiple threads
    }

    /// Called when optimization completes.
    ///
    /// In `InitialAndFinal` mode, this logs the final optimized state.
    /// In `Iterative` mode, the final state was already logged via `on_step`.
    ///
    /// # Arguments
    ///
    /// * `values` - Final optimized variable values
    /// * `iterations` - Total number of iterations performed
    fn on_optimization_complete(&self, values: &HashMap<String, VariableEnum>, iterations: usize) {
        if !self.is_enabled() {
            return;
        }

        // Log final state with displacement statistics
        if let Err(e) = self.log_final_state(values, iterations) {
            let _ = e.log();
        }
    }
}

impl Default for RerunObserver {
    fn default() -> Self {
        Self::new(false).unwrap_or_else(|_| Self {
            rec: None,
            enabled: false,
            iteration_metrics: RefCell::new(IterationMetrics::default()),
            config: VisualizationConfig::default(),
            initial_camera_positions: RefCell::new(HashMap::new()),
            initial_landmark_positions: RefCell::new(HashMap::new()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_observer_creation() -> TestResult {
        let observer = RerunObserver::new(false)?;
        assert!(!observer.is_enabled());
        Ok(())
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
    fn test_set_metrics() -> TestResult {
        let observer = RerunObserver::new(false)?;
        observer.set_iteration_metrics(1.0, 0.5, Some(0.01), 0.1, Some(0.95));

        let metrics = observer.iteration_metrics.borrow();
        assert_eq!(metrics.cost, Some(1.0));
        assert_eq!(metrics.gradient_norm, Some(0.5));
        assert_eq!(metrics.damping, Some(0.01));
        assert_eq!(metrics.step_norm, Some(0.1));
        assert_eq!(metrics.step_quality, Some(0.95));
        Ok(())
    }

    #[test]
    fn test_observer_trait() -> TestResult {
        let observer = RerunObserver::new(false)?;
        let values = HashMap::new();

        // Should not panic when disabled
        observer.on_step(&values, 0);
        observer.on_step(&values, 1);
        Ok(())
    }
}
