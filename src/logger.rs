//! Centralized logging configuration for apex-solver binaries and examples
//!
//! This module provides a consistent logging setup with custom formatting
//! and default INFO level across all executables.

use tracing::Level;

/// Initialize the tracing subscriber with apex-solver's standard configuration
///
/// Default log level: INFO (overrideable via RUST_LOG environment variable)
///
/// Format includes:
/// - Timestamp (YYYY-MM-DD HH:MM:SS)
/// - Log level (INFO, WARN, ERROR, DEBUG, TRACE)
/// - Module/target path
/// - File name and line number
///
/// # Example
/// ```no_run
/// use apex_solver::init_logger;
///
/// fn main() {
///     init_logger();
///     tracing::info!("Application started");
/// }
/// ```
///
/// # Environment Variables
/// Override the default log level using `RUST_LOG`:
/// ```bash
/// RUST_LOG=debug cargo run --bin optimize_3d_graph
/// RUST_LOG=apex_solver=trace cargo run
/// ```
pub fn init_logger() {
    init_logger_with_level(Level::INFO)
}

/// Initialize the tracing subscriber with a custom default level
///
/// # Arguments
/// * `default_level` - The default log level (overrideable via RUST_LOG)
///
/// # Example
/// ```no_run
/// use apex_solver::init_logger_with_level;
/// use tracing::Level;
///
/// fn main() {
///     init_logger_with_level(Level::DEBUG);
///     tracing::debug!("Debug logging enabled");
/// }
/// ```
pub fn init_logger_with_level(default_level: Level) {
    use tracing_subscriber::fmt::time::SystemTime;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(default_level.into())
                .from_env_lossy(),
        )
        // Custom format: [LEVEL: date:time module file:line]
        .with_timer(SystemTime) // YYYY-MM-DD HH:MM:SS format
        .with_target(true) // Include module path
        .with_level(true) // Include log level
        .with_file(false) // Include file name
        .with_line_number(false) // Include line number
        .with_thread_ids(false)
        .with_thread_names(false)
        .init();
}
