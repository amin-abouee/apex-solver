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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::rosbag::types::{
        Connection, MessageDefinition, QosDurability, QosReliability, RawMessage,
    };

    fn make_raw_message(
        topic: &str,
        msg_type: &str,
        timestamp_ns: u64,
        raw: Vec<u8>,
    ) -> RawMessage {
        RawMessage {
            connection: Connection {
                id: 1,
                topic: topic.to_string(),
                message_type: msg_type.to_string(),
                message_definition: MessageDefinition::default(),
                type_description_hash: String::new(),
                message_count: 0,
                serialization_format: "cdr".to_string(),
                offered_qos_profiles: Vec::new(),
            },
            timestamp: timestamp_ns,
            raw_data: raw,
        }
    }

    #[test]
    fn extract_header_stamp_too_short_returns_none() {
        assert!(extract_header_stamp(&[0x00, 0x01, 0x00]).is_none());
        assert!(extract_header_stamp(&[]).is_none());
    }

    #[test]
    fn extract_header_stamp_le_correct_value() {
        let mut data = [0u8; 12];
        data[1] = 0x01;
        data[4..8].copy_from_slice(&10i32.to_le_bytes());
        data[8..12].copy_from_slice(&500_000_000u32.to_le_bytes());
        let result = extract_header_stamp(&data).unwrap();
        assert!((result - 10.5).abs() < 1e-9);
    }

    #[test]
    fn extract_header_stamp_be_correct_value() {
        let mut data = [0u8; 12];
        data[1] = 0x00;
        data[4..8].copy_from_slice(&5i32.to_be_bytes());
        data[8..12].copy_from_slice(&0u32.to_be_bytes());
        let result = extract_header_stamp(&data).unwrap();
        assert!((result - 5.0).abs() < 1e-9);
    }

    #[test]
    fn extract_header_stamp_zero_returns_some_zero() {
        let data = [0u8; 12];
        let result = extract_header_stamp(&data).unwrap();
        assert!((result - 0.0).abs() < 1e-9);
    }

    #[test]
    fn received_message_recv_timestamp_conversion() {
        let raw = make_raw_message("/imu", "sensor_msgs/msg/Imu", 1_000_000_000, vec![0u8; 12]);
        let msg = ReceivedMessage::from_raw(raw);
        assert!((msg.recv_timestamp_s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn received_message_bytes_equals_raw_data_len() {
        let payload = vec![0x00u8; 20];
        let raw = make_raw_message("/imu", "sensor_msgs/msg/Imu", 0, payload.clone());
        let msg = ReceivedMessage::from_raw(raw);
        assert_eq!(msg.bytes, 20);
        assert_eq!(msg.raw_data, payload);
    }

    #[test]
    fn received_message_topic_and_type_forwarded() {
        let raw = make_raw_message("/cam", "sensor_msgs/msg/Image", 0, vec![]);
        let msg = ReceivedMessage::from_raw(raw);
        assert_eq!(msg.topic, "/cam");
        assert_eq!(msg.message_type, "sensor_msgs/msg/Image");
    }

    #[test]
    fn received_message_no_header_stamp_on_short_payload() {
        let raw = make_raw_message("/topic", "std_msgs/msg/String", 0, vec![0u8; 3]);
        let msg = ReceivedMessage::from_raw(raw);
        assert!(msg.msg_timestamp_s.is_none());
    }

    #[test]
    fn listener_new_stores_domain_id() {
        let listener = DdsListener::new(42);
        assert_eq!(listener.domain_id, 42);
    }

    #[test]
    fn listener_new_empty_subscriptions() {
        let listener = DdsListener::new(0);
        assert!(listener.subscriptions.is_empty());
    }

    #[test]
    fn listener_new_default_channel_capacity() {
        let listener = DdsListener::new(0);
        assert_eq!(listener.channel_capacity, 4096);
    }

    #[test]
    fn listener_subscribe_grows_subscriptions() {
        let listener = DdsListener::new(0)
            .subscribe("/imu", "sensor_msgs/msg/Imu", |_: ReceivedMessage| {})
            .subscribe("/gps", "sensor_msgs/msg/NavSatFix", |_: ReceivedMessage| {});
        assert_eq!(listener.subscriptions.len(), 2);
    }

    #[test]
    fn listener_subscribe_sets_config_fields() {
        let listener =
            DdsListener::new(7).subscribe("/cam", "sensor_msgs/msg/Image", |_: ReceivedMessage| {});
        let sub = &listener.subscriptions[0];
        assert_eq!(sub.config.topic, "/cam");
        assert_eq!(sub.config.message_type, "sensor_msgs/msg/Image");
        assert_eq!(sub.config.domain_id, 7);
        assert!(matches!(sub.config.reliability, QosReliability::BestEffort));
        assert!(matches!(sub.config.durability, QosDurability::Volatile));
    }

    #[tokio::test]
    async fn listener_spin_for_returns_immediately_with_no_subscriptions() {
        DdsListener::new(0)
            .spin_for(std::time::Duration::from_millis(1))
            .await;
    }

    #[tokio::test]
    async fn listener_spin_returns_immediately_with_no_subscriptions() {
        DdsListener::new(0).spin().await;
    }
}
