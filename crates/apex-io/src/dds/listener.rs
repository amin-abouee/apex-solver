use crate::dds::subscriber::{DdsSubscriber, DdsSubscriberConfig};
use crate::rosbag::types::{QosDurability, QosReliability, RawMessage};

// ── ReceivedMessage ───────────────────────────────────────────────────────────

/// A received DDS message, delivered to a subscription callback.
///
/// `msg_timestamp_s` holds the ROS2 `Header.stamp` extracted from the CDR
/// payload, if the message type carries a `std_msgs/Header` (e.g. IMU, GPS,
/// Image). For header-less types it is `None` and `recv_timestamp_s` (wall
/// clock) is the only timestamp available.
pub struct ReceivedMessage {
    /// ROS2 topic name (e.g. `/imu`).
    pub topic: String,
    /// ROS2 message type (e.g. `sensor_msgs/msg/Imu`).
    pub message_type: String,
    /// Sensor timestamp from `Header.stamp`, seconds since Unix epoch.
    pub msg_timestamp_s: Option<f64>,
    /// Wall-clock receive time, seconds since Unix epoch.
    pub recv_timestamp_s: f64,
    /// Payload size in bytes (CDR encapsulation header + body).
    pub bytes: usize,
    /// Full raw CDR payload — usable for bag writing or further parsing.
    pub raw_data: Vec<u8>,
}

impl ReceivedMessage {
    fn from_raw(raw: RawMessage) -> Self {
        let recv_timestamp_s = raw.timestamp as f64 / 1_000_000_000.0;
        let msg_timestamp_s = extract_header_stamp(&raw.raw_data);
        let bytes = raw.raw_data.len();
        Self {
            topic: raw.connection.topic.clone(),
            message_type: raw.connection.message_type.clone(),
            msg_timestamp_s,
            recv_timestamp_s,
            bytes,
            raw_data: raw.raw_data,
        }
    }
}

/// Parse `Header.stamp` out of a raw CDR buffer.
///
/// Layout after the 4-byte CDR encapsulation header:
///   bytes [4..8]  → stamp.sec  (i32)
///   bytes [8..12] → stamp.nanosec (u32)
/// Byte [1] == 0x01 means little-endian.
fn extract_header_stamp(data: &[u8]) -> Option<f64> {
    if data.len() < 12 {
        return None;
    }
    let le = data[1] == 0x01;
    let sec: i32 = if le {
        i32::from_le_bytes(data[4..8].try_into().ok()?)
    } else {
        i32::from_be_bytes(data[4..8].try_into().ok()?)
    };
    let nsec: u32 = if le {
        u32::from_le_bytes(data[8..12].try_into().ok()?)
    } else {
        u32::from_be_bytes(data[8..12].try_into().ok()?)
    };
    Some(sec as f64 + nsec as f64 / 1_000_000_000.0)
}

// ── DdsListener ───────────────────────────────────────────────────────────────

struct Subscription {
    config: DdsSubscriberConfig,
    callback: Box<dyn FnMut(ReceivedMessage) + Send + 'static>,
}

/// Subscribe to one or more ROS2 DDS topics and dispatch messages to
/// per-topic callbacks.
///
/// Build a listener with [`subscribe`](DdsListener::subscribe), then run it
/// with [`spin`](DdsListener::spin) (until Ctrl-C) or
/// [`spin_for`](DdsListener::spin_for) (for a fixed duration).
///
/// Each topic runs in its own background task; callbacks are called
/// sequentially within that topic but all topics are served concurrently.
///
/// # Example
/// ```no_run
/// use apex_io::dds::{DdsListener, ReceivedMessage};
///
/// #[tokio::main]
/// async fn main() {
///     DdsListener::new(0)
///         .subscribe("/imu", "sensor_msgs/msg/Imu", |msg: ReceivedMessage| {
///             tracing::info!(ts = ?msg.msg_timestamp_s, bytes = msg.bytes, "IMU");
///         })
///         .subscribe("/gps", "sensor_msgs/msg/NavSatFix", |msg: ReceivedMessage| {
///             tracing::info!(ts = ?msg.msg_timestamp_s, bytes = msg.bytes, "GPS");
///         })
///         .spin()
///         .await;
/// }
/// ```
pub struct DdsListener {
    domain_id: u16,
    channel_capacity: usize,
    subscriptions: Vec<Subscription>,
}

impl DdsListener {
    /// Create a listener on the given DDS domain ID.
    ///
    /// Pass the value of the `ROS_DOMAIN_ID` environment variable (default `0`)
    /// to match the running ROS2 system.
    pub fn new(domain_id: u16) -> Self {
        Self {
            domain_id,
            channel_capacity: 4096,
            subscriptions: Vec::new(),
        }
    }

    /// Register a callback for a ROS2 topic (builder-style).
    ///
    /// Uses best-effort / volatile QoS, which matches the default profile
    /// used by `ros2 bag play`.
    pub fn subscribe<F>(
        mut self,
        topic: impl Into<String>,
        message_type: impl Into<String>,
        callback: F,
    ) -> Self
    where
        F: FnMut(ReceivedMessage) + Send + 'static,
    {
        let config = DdsSubscriberConfig {
            topic: topic.into(),
            message_type: message_type.into(),
            reliability: QosReliability::BestEffort,
            durability: QosDurability::Volatile,
            history_depth: 10,
            domain_id: self.domain_id,
            channel_capacity: self.channel_capacity,
        };
        self.subscriptions.push(Subscription {
            config,
            callback: Box::new(callback),
        });
        self
    }

    /// Run indefinitely, calling each registered callback as messages arrive.
    ///
    /// Returns when Ctrl-C is received.
    pub async fn spin(self) {
        self.run(None).await;
    }

    /// Run for a fixed duration, calling each registered callback as messages arrive.
    pub async fn spin_for(self, duration: std::time::Duration) {
        self.run(Some(duration)).await;
    }

    async fn run(self, duration: Option<std::time::Duration>) {
        // Start one tokio task per topic.
        let handles: Vec<_> = self
            .subscriptions
            .into_iter()
            .filter_map(|sub| {
                let topic = sub.config.topic.clone();
                let mut callback = sub.callback;

                let rx = match DdsSubscriber::new(sub.config).and_then(|s| s.listen()) {
                    Ok(rx) => rx,
                    Err(e) => {
                        tracing::error!(topic = %topic, error = %e, "Failed to start subscriber");
                        return None;
                    }
                };

                Some(tokio::spawn(async move {
                    let mut rx = rx;
                    while let Some(raw_msg) = rx.recv().await {
                        callback(ReceivedMessage::from_raw(raw_msg));
                    }
                }))
            })
            .collect();

        if handles.is_empty() {
            tracing::warn!("No subscriptions could be started");
            return;
        }

        // Wait for a stop signal, then abort all tasks.
        tokio::select! {
            _ = async {
                if let Some(d) = duration {
                    tokio::time::sleep(d).await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {
                tracing::info!("Duration elapsed, stopping listener");
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Ctrl-C received, stopping listener");
            }
        }

        for handle in handles {
            handle.abort();
        }
    }
}
