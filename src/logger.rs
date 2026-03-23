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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::fmt::MakeWriter;

    struct SharedBuf(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[derive(Clone)]
    struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

    impl CapturingWriter {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Vec::new())))
        }

        fn captured(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().unwrap_or_else(|e| e.into_inner()))
                .to_string()
        }
    }

    impl<'a> MakeWriter<'a> for CapturingWriter {
        type Writer = SharedBuf;

        fn make_writer(&'a self) -> Self::Writer {
            SharedBuf(Arc::clone(&self.0))
        }
    }

    fn capturing_subscriber(writer: CapturingWriter) -> impl tracing::Subscriber + Send + Sync {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::new("trace"))
            .with_writer(writer)
            .event_format(CustomFormatter)
            .finish()
    }

    #[test]
    fn info_output_contains_ansi_green() {
        let w = CapturingWriter::new();
        tracing::subscriber::with_default(capturing_subscriber(w.clone()), || {
            tracing::info!("hello info");
        });
        let out = w.captured();
        assert!(
            out.contains("\x1b[32mINFO\x1b[0m"),
            "expected green INFO ANSI code, got: {out:?}"
        );
    }

    #[test]
    fn warn_output_contains_ansi_yellow() {
        let w = CapturingWriter::new();
        tracing::subscriber::with_default(capturing_subscriber(w.clone()), || {
            tracing::warn!("hello warn");
        });
        let out = w.captured();
        assert!(
            out.contains("\x1b[33mWARN\x1b[0m"),
            "expected yellow WARN ANSI code, got: {out:?}"
        );
    }

    #[test]
    fn error_output_contains_ansi_red() {
        let w = CapturingWriter::new();
        tracing::subscriber::with_default(capturing_subscriber(w.clone()), || {
            tracing::error!("hello error");
        });
        let out = w.captured();
        assert!(
            out.contains("\x1b[31mERROR\x1b[0m"),
            "expected red ERROR ANSI code, got: {out:?}"
        );
    }

    #[test]
    fn debug_output_contains_ansi_blue_and_filename() {
        let w = CapturingWriter::new();
        tracing::subscriber::with_default(capturing_subscriber(w.clone()), || {
            tracing::debug!("hello debug");
        });
        let out = w.captured();
        assert!(
            out.contains("\x1b[34mDEBUG\x1b[0m"),
            "expected blue DEBUG ANSI code, got: {out:?}"
        );
        // DEBUG branch uses file:line — the filename should appear before ':'
        assert!(
            out.contains("logger.rs:"),
            "expected file:line in DEBUG output, got: {out:?}"
        );
    }

    #[test]
    fn trace_output_contains_ansi_magenta_and_filename() {
        let w = CapturingWriter::new();
        tracing::subscriber::with_default(capturing_subscriber(w.clone()), || {
            tracing::trace!("hello trace");
        });
        let out = w.captured();
        assert!(
            out.contains("\x1b[35mTRACE\x1b[0m"),
            "expected magenta TRACE ANSI code, got: {out:?}"
        );
        assert!(
            out.contains("logger.rs:"),
            "expected file:line in TRACE output, got: {out:?}"
        );
    }

    #[test]
    fn output_contains_timestamp() {
        let w = CapturingWriter::new();
        tracing::subscriber::with_default(capturing_subscriber(w.clone()), || {
            tracing::info!("timestamp test");
        });
        let out = w.captured();
        // Timestamp format: YYYY-MM-DD HH:MM:SS — check for year prefix " 20"
        assert!(
            out.contains(" 20"),
            "expected timestamp starting with year 20xx, got: {out:?}"
        );
        // Date contains hyphens and time contains colons
        assert!(
            out.contains('-'),
            "expected date separator '-', got: {out:?}"
        );
        assert!(
            out.contains(':'),
            "expected time separator ':', got: {out:?}"
        );
    }

    #[test]
    fn output_ends_with_newline() {
        let w = CapturingWriter::new();
        tracing::subscriber::with_default(capturing_subscriber(w.clone()), || {
            tracing::info!("newline test");
        });
        let out = w.captured();
        assert!(
            out.ends_with('\n'),
            "expected trailing newline, got: {out:?}"
        );
    }

    #[test]
    fn output_contains_message_text() {
        let w = CapturingWriter::new();
        tracing::subscriber::with_default(capturing_subscriber(w.clone()), || {
            tracing::info!("unique_sentinel_abc123");
        });
        let out = w.captured();
        assert!(
            out.contains("unique_sentinel_abc123"),
            "expected message in output, got: {out:?}"
        );
    }

    #[test]
    fn output_is_wrapped_in_brackets() {
        let w = CapturingWriter::new();
        tracing::subscriber::with_default(capturing_subscriber(w.clone()), || {
            tracing::info!("bracket test");
        });
        let out = w.captured();
        assert!(out.starts_with('['), "expected '[' at start, got: {out:?}");
        assert!(
            out.contains("] "),
            "expected '] ' closing bracket, got: {out:?}"
        );
    }

    #[test]
    fn info_uses_target_not_file_line() {
        let w = CapturingWriter::new();
        tracing::subscriber::with_default(capturing_subscriber(w.clone()), || {
            tracing::info!("info target check");
        });
        let out = w.captured();
        // INFO/WARN/ERROR show module target, not file:line
        assert!(
            !out.contains("logger.rs:"),
            "INFO output must use target, not file:line, got: {out:?}"
        );
    }

    // ---------------------------------------------------------------------------
    // Smoke test for the public init API
    // ---------------------------------------------------------------------------

    #[test]
    fn init_logger_with_level_does_not_panic() {
        // Can't call the real init_logger() (sets global subscriber once).
        // Verify the builder path compiles and runs via try_init.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::new("off"))
            .event_format(CustomFormatter)
            .try_init();
    }
}
