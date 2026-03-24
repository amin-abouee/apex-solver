//! Centralized logging configuration for apex-io binaries.
//!
//! Mirrors the setup in `apex-solver`'s logger: default INFO level,
//! custom `[LEVEL YYYY-MM-DD HH:MM:SS target]` format, and respects `RUST_LOG`.

use tracing::Level;

/// Initialize the tracing subscriber with the standard apex-io configuration.
///
/// Default log level: INFO (overrideable via `RUST_LOG`).
///
/// Format: `[LEVEL YYYY-MM-DD HH:MM:SS module]`
pub fn init_logger() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(Level::INFO.into())
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

/// Custom event formatter matching apex-solver's bracketed style.
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

        let metadata = event.metadata();
        let level = metadata.level();

        write!(writer, "[")?;
        match *level {
            Level::ERROR => write!(writer, "\x1b[31mERROR\x1b[0m ")?,
            Level::WARN => write!(writer, "\x1b[33mWARN\x1b[0m ")?,
            Level::INFO => write!(writer, "\x1b[32mINFO\x1b[0m ")?,
            Level::DEBUG => write!(writer, "\x1b[34mDEBUG\x1b[0m ")?,
            Level::TRACE => write!(writer, "\x1b[35mTRACE\x1b[0m ")?,
        }

        write!(writer, "{} ", Local::now().format("%Y-%m-%d %H:%M:%S"))?;

        if *level == Level::DEBUG || *level == Level::TRACE {
            if let Some(file) = metadata.file() {
                let filename = file.rsplit('/').next().unwrap_or(file);
                write!(writer, "{}:", filename)?;
                if let Some(line) = metadata.line() {
                    write!(writer, "{line}")?;
                }
            } else {
                write!(writer, "{}", metadata.target())?;
            }
        } else {
            write!(writer, "{}", metadata.target())?;
        }

        write!(writer, "] ")?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}
