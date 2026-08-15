//! Listen to multiple ROS2 DDS topics simultaneously and log each message.
//!
//! Reads `ROS_DOMAIN_ID` from the environment (default 0).
//! Run while a bag is playing:
//!
//!   ros2 bag play <path/to/bag>
//!   cargo run -p apex-io --features dds --bin dds_multi_listener

use apex_io::dds::{DdsListener, ReceivedMessage};

// ── per-type named callbacks ──────────────────────────────────────────────────
// Each function receives the per-topic message counter as its first argument.

/// Log every message at debug level, plus a summary at info every
/// `SUMMARY_EVERY` messages to keep the default log readable at high rates.
const SUMMARY_EVERY: u64 = 100;

fn log_message(tag: &'static str, count: u64, msg: &ReceivedMessage) {
    tracing::debug!(
        count,
        topic           = %msg.topic,
        msg_timestamp_s = ?msg.msg_timestamp_s,
        bytes           = msg.bytes,
        "{tag}"
    );
    if count == 1 || count % SUMMARY_EVERY == 0 {
        tracing::info!(
            count,
            topic           = %msg.topic,
            msg_timestamp_s = ?msg.msg_timestamp_s,
            bytes           = msg.bytes,
            "{tag}"
        );
    }
}

fn on_imu(count: u64, msg: ReceivedMessage) {
    log_message("IMU", count, &msg);
}

fn on_leica_position(count: u64, msg: ReceivedMessage) {
    log_message("Position", count, &msg);
}

fn on_image(count: u64, msg: ReceivedMessage) {
    log_message("Image", count, &msg);
}

// ── counted wrapper ───────────────────────────────────────────────────────────

/// Wraps a `fn(count, msg)` callback and maintains its own per-topic counter.
///
/// Each call to the returned closure increments the counter by one and
/// forwards it to `f`.  Because the counter is owned by the closure, topics
/// are counted independently with no shared state.
fn counted<F>(mut f: F) -> impl FnMut(ReceivedMessage) + Send + 'static
where
    F: FnMut(u64, ReceivedMessage) + Send + 'static,
{
    let mut n: u64 = 0;
    move |msg| {
        n += 1;
        f(n, msg);
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,rustdds=off")),
        )
        .init();

    let domain_id: u16 = std::env::var("ROS_DOMAIN_ID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    tracing::info!(
        domain_id,
        "Starting multi-topic DDS listener (Ctrl-C to stop)"
    );

    DdsListener::new(domain_id)
        .subscribe(
            "/leica/position",
            "geometry_msgs/msg/PointStamped",
            counted(on_leica_position),
        )
        .subscribe("/imu0", "sensor_msgs/msg/Imu", counted(on_imu))
        .subscribe(
            "/cam0/image_raw",
            "sensor_msgs/msg/Image",
            counted(on_image),
        )
        .spin()
        .await;
}
