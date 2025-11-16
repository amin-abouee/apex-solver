//! Observer pattern for optimization monitoring.
//!
//! This module provides a clean observer pattern for monitoring optimization progress.
//! Observers can be registered with any optimizer and will be notified at each iteration,
//! enabling real-time visualization, logging, metrics collection, and custom analysis.
//!
//! # Design Philosophy
//!
//! The observer pattern provides complete separation between optimization algorithms
//! and monitoring/visualization logic:
//!
//! - **Decoupling**: Optimization logic is independent of how progress is monitored
//! - **Extensibility**: Easy to add new observers (Rerun, CSV, metrics, dashboards)
//! - **Composability**: Multiple observers can run simultaneously
//! - **Zero overhead**: When no observers are registered, notification is a no-op
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐
//! │   Optimizer     │
//! │  (LM/GN/DogLeg) │
//! └────────┬────────┘
//!          │ observers.notify(values, iteration)
//!          ├──────────────┬──────────────┬──────────────┐
//!          ▼              ▼              ▼              ▼
//!    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
//!    │  Rerun   │  │   CSV    │  │ Metrics  │  │  Custom  │
//!    │ Observer │  │ Observer │  │ Observer │  │ Observer │
//!    └──────────┘  └──────────┘  └──────────┘  └──────────┘
//! ```
//!
//! # Examples
//!
//! ## Single Observer
//!
//! ```no_run
//! use apex_solver::{LevenbergMarquardt, LevenbergMarquardtConfig};
//! use apex_solver::observers::OptObserver;
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
//! #[cfg(feature = "visualization")]
//! {
//!     use apex_solver::observers::RerunObserver;
//!     let rerun_observer = RerunObserver::new(true)?;
//!     solver.add_observer(rerun_observer);
//! }
//!
//! let result = solver.optimize(&problem, &initial_values)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Multiple Observers
//!
//! ```no_run
//! # use apex_solver::{LevenbergMarquardt, LevenbergMarquardtConfig};
//! # use apex_solver::core::problem::{Problem, VariableEnum};
//! # use apex_solver::observers::OptObserver;
//! # use std::collections::HashMap;
//!
//! // Custom observer that logs to CSV
//! struct CsvObserver {
//!     file: std::fs::File,
//! }
//!
//! impl OptObserver for CsvObserver {
//!     fn on_step(&self, _values: &HashMap<String, VariableEnum>, iteration: usize) {
//!         // Write iteration data to CSV
//!         // ... implementation ...
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let problem = Problem::new();
//! # let initial_values = HashMap::new();
//! let mut solver = LevenbergMarquardt::new();
//!
//! // Add Rerun visualization
//! #[cfg(feature = "visualization")]
//! {
//!     use apex_solver::observers::RerunObserver;
//!     solver.add_observer(RerunObserver::new(true)?);
//! }
//!
//! // Add CSV logging
//! // solver.add_observer(CsvObserver { file: ... });
//!
//! let result = solver.optimize(&problem, &initial_values)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Observer
//!
//! ```no_run
//! use apex_solver::observers::OptObserver;
//! use apex_solver::core::problem::VariableEnum;
//! use std::collections::HashMap;
//!
//! struct MetricsObserver {
//!     max_variables_seen: std::cell::RefCell<usize>,
//! }
//!
//! impl OptObserver for MetricsObserver {
//!     fn on_step(&self, values: &HashMap<String, VariableEnum>, iteration: usize) {
//!         let count = values.len();
//!         let mut max = self.max_variables_seen.borrow_mut();
//!         *max = (*max).max(count);
//!
//!         println!("Iteration {}: {} variables", iteration, count);
//!     }
//! }
//! ```

// Visualization-specific submodules (feature-gated)
#[cfg(feature = "visualization")]
pub mod conversions;
#[cfg(feature = "visualization")]
pub mod visualization;

// Re-export RerunObserver when visualization is enabled
#[cfg(feature = "visualization")]
pub use visualization::RerunObserver;

use crate::core::problem::VariableEnum;
use faer::Mat;
use faer::sparse;
use std::collections::HashMap;

/// Observer trait for monitoring optimization progress.
///
/// Implement this trait to create custom observers that are notified at each
/// optimization iteration. Observers receive the current variable values and
/// iteration number, enabling real-time monitoring, visualization, logging,
/// or custom analysis.
///
/// # Design Notes
///
/// - Observers should be lightweight and non-blocking
/// - Errors in observers should not crash optimization (handle internally)
/// - For expensive operations (file I/O, network), consider buffering
/// - Observers receive immutable references (cannot modify optimization state)
///
/// # Thread Safety
///
/// Observers must be `Send` to support parallel optimization in the future.
/// Use interior mutability (`RefCell`, `Mutex`) if you need to mutate state.
pub trait OptObserver: Send {
    /// Called after each optimization iteration.
    ///
    /// # Arguments
    ///
    /// * `values` - Current variable values (manifold states)
    /// * `iteration` - Current iteration number (0 = initial values, 1+ = after steps)
    ///
    /// # Implementation Guidelines
    ///
    /// - Keep this method fast to avoid slowing optimization
    /// - Handle errors internally (log warnings, don't panic)
    /// - Don't mutate `values` (you receive `&HashMap`)
    /// - Consider buffering expensive operations
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::observers::OptObserver;
    /// use apex_solver::core::problem::VariableEnum;
    /// use std::collections::HashMap;
    ///
    /// struct SimpleLogger;
    ///
    /// impl OptObserver for SimpleLogger {
    ///     fn on_step(&self, values: &HashMap<String, VariableEnum>, iteration: usize) {
    ///         println!("Iteration {}: {} variables", iteration, values.len());
    ///     }
    /// }
    /// ```
    fn on_step(&self, values: &HashMap<String, VariableEnum>, iteration: usize);

    /// Set iteration metrics for visualization and monitoring.
    ///
    /// This method is called before `on_step` to provide optimization metrics
    /// such as cost, gradient norm, damping parameter, etc. Observers can use
    /// this data for visualization, logging, or analysis.
    ///
    /// # Arguments
    ///
    /// * `cost` - Current cost function value
    /// * `gradient_norm` - L2 norm of the gradient vector
    /// * `damping` - Damping parameter (for Levenberg-Marquardt, may be None for other solvers)
    /// * `step_norm` - L2 norm of the parameter update step
    /// * `step_quality` - Step quality metric (e.g., rho for trust region methods)
    ///
    /// # Default Implementation
    ///
    /// The default implementation does nothing, allowing simple observers to ignore metrics.
    fn set_iteration_metrics(
        &self,
        _cost: f64,
        _gradient_norm: f64,
        _damping: Option<f64>,
        _step_norm: f64,
        _step_quality: Option<f64>,
    ) {
        // Default implementation does nothing
    }

    /// Set matrix data for advanced visualization.
    ///
    /// This method provides access to the Hessian matrix and gradient vector
    /// for observers that want to visualize matrix structure or perform
    /// advanced analysis.
    ///
    /// # Arguments
    ///
    /// * `hessian` - Sparse Hessian matrix (J^T * J)
    /// * `gradient` - Gradient vector (J^T * r)
    ///
    /// # Default Implementation
    ///
    /// The default implementation does nothing, allowing simple observers to ignore matrices.
    fn set_matrix_data(
        &self,
        _hessian: Option<sparse::SparseColMat<usize, f64>>,
        _gradient: Option<Mat<f64>>,
    ) {
        // Default implementation does nothing
    }
}

/// Collection of observers for optimization monitoring.
///
/// This struct manages a vector of observers and provides a convenient
/// `notify()` method to call all observers at once. Optimizers use this
/// internally to manage their observers.
///
/// # Usage
///
/// Typically you don't create this directly - use the `add_observer()` method
/// on optimizers. However, you can use it for custom optimization algorithms:
///
/// ```no_run
/// use apex_solver::observers::{OptObserver, OptObserverVec};
/// use apex_solver::core::problem::VariableEnum;
/// use std::collections::HashMap;
///
/// struct MyOptimizer {
///     observers: OptObserverVec,
///     // ... other fields ...
/// }
///
/// impl MyOptimizer {
///     fn step(&mut self, values: &HashMap<String, VariableEnum>, iteration: usize) {
///         // ... optimization logic ...
///
///         // Notify all observers
///         self.observers.notify(values, iteration);
///     }
/// }
/// ```
#[derive(Default)]
pub struct OptObserverVec {
    observers: Vec<Box<dyn OptObserver>>,
}

impl OptObserverVec {
    /// Create a new empty observer collection.
    pub fn new() -> Self {
        Self {
            observers: Vec::new(),
        }
    }

    /// Add an observer to the collection.
    ///
    /// The observer will be called at each optimization iteration in the order
    /// it was added.
    ///
    /// # Arguments
    ///
    /// * `observer` - Any type implementing `OptObserver`
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::observers::{OptObserver, OptObserverVec};
    /// use apex_solver::core::problem::VariableEnum;
    /// use std::collections::HashMap;
    ///
    /// struct MyObserver;
    /// impl OptObserver for MyObserver {
    ///     fn on_step(&self, _values: &HashMap<String, VariableEnum>, _iteration: usize) {
    ///         println!("Optimization step!");
    ///     }
    /// }
    ///
    /// let mut observers = OptObserverVec::new();
    /// observers.add(MyObserver);
    /// ```
    pub fn add(&mut self, observer: impl OptObserver + 'static) {
        self.observers.push(Box::new(observer));
    }

    /// Set iteration metrics for all observers.
    ///
    /// Calls `set_iteration_metrics()` on each registered observer. This should
    /// be called before `notify()` to provide optimization metrics.
    ///
    /// # Arguments
    ///
    /// * `cost` - Current cost function value
    /// * `gradient_norm` - L2 norm of the gradient vector
    /// * `damping` - Damping parameter (may be None)
    /// * `step_norm` - L2 norm of the parameter update step
    /// * `step_quality` - Step quality metric (may be None)
    #[inline]
    pub fn set_iteration_metrics(
        &self,
        cost: f64,
        gradient_norm: f64,
        damping: Option<f64>,
        step_norm: f64,
        step_quality: Option<f64>,
    ) {
        for observer in &self.observers {
            observer.set_iteration_metrics(cost, gradient_norm, damping, step_norm, step_quality);
        }
    }

    /// Set matrix data for all observers.
    ///
    /// Calls `set_matrix_data()` on each registered observer. This should
    /// be called before `notify()` to provide matrix data for visualization.
    ///
    /// # Arguments
    ///
    /// * `hessian` - Sparse Hessian matrix
    /// * `gradient` - Gradient vector
    #[inline]
    pub fn set_matrix_data(
        &self,
        hessian: Option<sparse::SparseColMat<usize, f64>>,
        gradient: Option<Mat<f64>>,
    ) {
        for observer in &self.observers {
            observer.set_matrix_data(hessian.clone(), gradient.clone());
        }
    }

    /// Notify all observers with current optimization state.
    ///
    /// Calls `on_step()` on each registered observer in order. If no observers
    /// are registered, this is a no-op with zero overhead.
    ///
    /// # Arguments
    ///
    /// * `values` - Current variable values
    /// * `iteration` - Current iteration number
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::observers::OptObserverVec;
    /// use std::collections::HashMap;
    ///
    /// let observers = OptObserverVec::new();
    /// let values = HashMap::new();
    ///
    /// // Notify all observers (safe even if empty)
    /// observers.notify(&values, 0);
    /// ```
    #[inline]
    pub fn notify(&self, values: &HashMap<String, VariableEnum>, iteration: usize) {
        for observer in &self.observers {
            observer.on_step(values, iteration);
        }
    }

    /// Check if any observers are registered.
    ///
    /// Useful for conditional logic or debugging.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.observers.is_empty()
    }

    /// Get the number of registered observers.
    #[inline]
    pub fn len(&self) -> usize {
        self.observers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct TestObserver {
        calls: Arc<Mutex<Vec<usize>>>,
    }

    impl OptObserver for TestObserver {
        fn on_step(&self, _values: &HashMap<String, VariableEnum>, iteration: usize) {
            self.calls.lock().unwrap().push(iteration);
        }
    }

    #[test]
    fn test_empty_observers() {
        let observers = OptObserverVec::new();
        assert!(observers.is_empty());
        assert_eq!(observers.len(), 0);

        // Should not panic with no observers
        observers.notify(&HashMap::new(), 0);
    }

    #[test]
    fn test_single_observer() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observer = TestObserver {
            calls: calls.clone(),
        };

        let mut observers = OptObserverVec::new();
        observers.add(observer);

        assert_eq!(observers.len(), 1);

        observers.notify(&HashMap::new(), 0);
        observers.notify(&HashMap::new(), 1);
        observers.notify(&HashMap::new(), 2);

        assert_eq!(*calls.lock().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn test_multiple_observers() {
        let calls1 = Arc::new(Mutex::new(Vec::new()));
        let calls2 = Arc::new(Mutex::new(Vec::new()));

        let observer1 = TestObserver {
            calls: calls1.clone(),
        };
        let observer2 = TestObserver {
            calls: calls2.clone(),
        };

        let mut observers = OptObserverVec::new();
        observers.add(observer1);
        observers.add(observer2);

        assert_eq!(observers.len(), 2);

        observers.notify(&HashMap::new(), 5);

        assert_eq!(*calls1.lock().unwrap(), vec![5]);
        assert_eq!(*calls2.lock().unwrap(), vec![5]);
    }
}
