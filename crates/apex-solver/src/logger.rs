//! Centralized logging configuration for apex-solver binaries and examples
//!
//! This module provides a consistent logging setup with custom formatting
//! and default INFO level across all executables.

use tracing::Level;

/// Initialize the tracing subscriber with apex-solver's standard configuration
///
/// Default log level: INFO (overrideable via RUST_LOG environment variable)
///
/// Format: `[LEVEL YYYY-MM-DD HH:MM:SS module]` for INFO/WARN/ERROR
///         `[LEVEL YYYY-MM-DD HH:MM:SS file:line]` for DEBUG/TRACE
///
/// # Example
/// ```no_run
/// use apex_solver::init_logger;
///
/// init_logger();
/// tracing::info!("Application started");
/// ```
///
/// # Environment Variables
/// Override the default log level using `RUST_LOG`:
/// ```bash
/// RUST_LOG=debug cargo run --bin pose_graph_g2o
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
/// init_logger_with_level(Level::DEBUG);
/// tracing::debug!("Debug logging enabled");
/// ```
pub fn init_logger_with_level(default_level: Level) {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(default_level.into())
                .from_env_lossy(),
        )
        .with_target(false)
        .with_level(false)
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .event_format(CustomFormatter)
        .init();
}

/// Custom event formatter for cleaner output with brackets
struct CustomFormatter;

impl<S, N> tracing_subscriber::fmt::FormatEvent<S, N> for CustomFormatter
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> tracing_subscriber::fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: tracing_subscriber::fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        use chrono::Local;

        // Get metadata
        let metadata = event.metadata();
        let level = metadata.level();

        // Format: [LEVEL: YYYY-MM-DD HH:MM:SS module] message
        // For DEBUG/TRACE: [LEVEL: YYYY-MM-DD HH:MM:SS file:line] message
        write!(writer, "[")?;

        // Colored level
        match *level {
            Level::ERROR => write!(writer, "\x1b[31mERROR\x1b[0m ")?,
            Level::WARN => write!(writer, "\x1b[33mWARN\x1b[0m ")?,
            Level::INFO => write!(writer, "\x1b[32mINFO\x1b[0m ")?,
            Level::DEBUG => write!(writer, "\x1b[34mDEBUG\x1b[0m ")?,
            Level::TRACE => write!(writer, "\x1b[35mTRACE\x1b[0m ")?,
        }

        // Timestamp
        write!(writer, "{} ", Local::now().format("%Y-%m-%d %H:%M:%S"))?;

        // For DEBUG/TRACE, show file:line; for INFO/WARN/ERROR, show module
        if *level == Level::DEBUG || *level == Level::TRACE {
            if let Some(file) = metadata.file() {
                // Extract just the filename without path
                let filename = file.rsplit('/').next().unwrap_or(file);
                write!(writer, "{}:", filename)?;
                if let Some(line) = metadata.line() {
                    write!(writer, "{}", line)?;
                }
            } else {
                write!(writer, "{}", metadata.target())?;
            }
        } else {
            // For INFO/WARN/ERROR, show module name
            write!(writer, "{}", metadata.target())?;
        }

        write!(writer, "] ")?;

        // Write the message
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}
