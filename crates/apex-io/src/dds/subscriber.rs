use crate::dds::error::{DdsError, Result};
use crate::dds::qos_mapping::{to_dds_durability, to_dds_history, to_dds_reliability};
use crate::dds::raw_bytes::{RawBytes, RawBytesAdapter};
use crate::rosbag::types::{
    Connection, MessageDefinition, QosDurability, QosReliability, RawMessage,
};
use futures::StreamExt;
use rustdds::{DomainParticipant, QosPolicyBuilder, TopicKind};
use tokio::sync::mpsc;

/// Configuration for a DDS subscriber.
pub struct DdsSubscriberConfig {
    pub topic: String,
    pub message_type: String,
    pub reliability: QosReliability,
    pub durability: QosDurability,
    pub history_depth: i32,
    pub domain_id: u16,
    pub channel_capacity: usize,
}

impl Default for DdsSubscriberConfig {
    fn default() -> Self {
        Self {
            topic: String::new(),
            message_type: String::new(),
            reliability: QosReliability::BestEffort,
            durability: QosDurability::Volatile,
            history_depth: 10,
            domain_id: 0,
            channel_capacity: 4096,
        }
    }
}

/// Async DDS subscriber that delivers `RawMessage`s over a Tokio mpsc channel.
pub struct DdsSubscriber {
    config: DdsSubscriberConfig,
}

impl DdsSubscriber {
    pub fn new(config: DdsSubscriberConfig) -> Result<Self> {
        if config.topic.is_empty() {
            return Err(DdsError::InvalidTopicName {
                name: config.topic.clone(),
                reason: "topic name must not be empty".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Maps a ROS2 topic name (e.g. `/imu`) to its DDS wire name (e.g. `rt/imu`).
    pub fn ros2_to_dds_topic(ros_topic: &str) -> String {
        let stripped = ros_topic.trim_start_matches('/');
        if stripped.starts_with("rt/") {
            stripped.to_string()
        } else {
            format!("rt/{stripped}")
        }
    }

    /// Maps a ROS2 type name to its DDS wire type name.
    ///
    /// ROS2 DDS type name mangling: `sensor_msgs/msg/Imu` → `sensor_msgs::msg::dds_::Imu_`
    pub fn ros2_type_to_dds_type(ros2_type: &str) -> String {
        let parts: Vec<&str> = ros2_type.splitn(3, '/').collect();
        if parts.len() == 3 {
            format!("{}::{}::dds_::{}_", parts[0], parts[1], parts[2])
        } else {
            ros2_type.to_string()
        }
    }

    /// Start listening. Returns a channel receiver delivering `RawMessage`s.
    ///
    /// Spawns a dedicated OS thread that drives the rustdds async stream loop.
    /// The thread sends messages over the returned receiver. Dropping the receiver
    /// signals the thread to shut down gracefully.
    pub fn listen(self) -> Result<mpsc::Receiver<RawMessage>> {
        let (tx, rx) = mpsc::channel::<RawMessage>(self.config.channel_capacity);
        let config = self.config;
        let dds_topic_name = Self::ros2_to_dds_topic(&config.topic);

        std::thread::Builder::new()
            .name(format!("dds-{}", config.topic))
            .spawn(move || {
                if let Err(e) = run_reader_loop(config, dds_topic_name, tx) {
                    tracing::error!("DDS reader loop exited with error: {e}");
                }
            })
            .map_err(|e| DdsError::ThreadJoin(e.to_string()))?;

        Ok(rx)
    }
}

fn run_reader_loop(
    config: DdsSubscriberConfig,
    dds_topic_name: String,
    tx: mpsc::Sender<RawMessage>,
) -> Result<()> {
    let reliability = to_dds_reliability(&config.reliability);
    let durability = to_dds_durability(&config.durability);
    let history = to_dds_history(config.history_depth);

    let qos = QosPolicyBuilder::new()
        .reliability(reliability)
        .durability(durability)
        .history(history)
        .build();

    let participant = DomainParticipant::new(config.domain_id)
        .map_err(|e| DdsError::ParticipantCreation(e.to_string()))?;

    let dds_type_name = DdsSubscriber::ros2_type_to_dds_type(&config.message_type);

    tracing::info!(
        dds_topic = %dds_topic_name,
        dds_type = %dds_type_name,
        domain_id = config.domain_id,
        "Creating DDS topic"
    );

    let topic = participant
        .create_topic(
            dds_topic_name.clone(),
            dds_type_name,
            &qos,
            TopicKind::NoKey,
        )
        .map_err(|e| DdsError::TopicCreation {
            topic: dds_topic_name.clone(),
            reason: e.to_string(),
        })?;

    let subscriber = participant
        .create_subscriber(&qos)
        .map_err(|e| DdsError::SubscriberCreation(e.to_string()))?;

    let reader = subscriber
        .create_datareader_no_key::<RawBytes, RawBytesAdapter>(&topic, Some(qos))
        .map_err(|e| DdsError::DataReaderCreation(e.to_string()))?;

    let connection = Connection {
        id: 1,
        topic: config.topic.clone(),
        message_type: config.message_type.clone(),
        message_definition: MessageDefinition::default(),
        type_description_hash: String::new(),
        message_count: 0,
        serialization_format: "cdr".to_string(),
        offered_qos_profiles: Vec::new(),
    };

    // Run the async stream loop on a single-threaded tokio runtime in this OS thread.
    // rustdds stores the waker and calls it from its internal protocol thread when
    // data arrives, so there is no busy-wait.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| DdsError::ThreadJoin(e.to_string()))?;

    rt.block_on(async move {
        use tokio::sync::mpsc::error::TrySendError;

        let mut stream = reader.async_sample_stream();
        let mut dropped: u64 = 0;

        loop {
            match stream.next().await {
                Some(Ok(RawBytes(raw_data))) => {
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0);

                    let msg = RawMessage {
                        connection: connection.clone(),
                        timestamp,
                        raw_data,
                    };

                    // Non-blocking send: if the consumer is too slow we drop
                    // one message rather than stalling the DDS reader (which
                    // would cause the kernel UDP socket to overflow and lose
                    // far more fragments).
                    match tx.try_send(msg) {
                        Ok(()) => {}
                        Err(TrySendError::Full(_)) => {
                            dropped = dropped.saturating_add(1);
                            if dropped.is_power_of_two() {
                                tracing::warn!(
                                    topic = %config.topic,
                                    dropped,
                                    "consumer slow — dropped message (channel full)"
                                );
                            }
                        }
                        Err(TrySendError::Closed(_)) => {
                            tracing::debug!("DDS channel receiver dropped, shutting down reader");
                            return;
                        }
                    }
                }
                Some(Err(e)) => {
                    tracing::warn!("DDS read error: {e}");
                }
                None => {
                    tracing::info!("DDS stream ended");
                    break;
                }
            }
        }
    });

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::rosbag::types::{QosDurability, QosReliability};

    #[test]
    fn ros2_to_dds_topic_strips_leading_slash() {
        assert_eq!(DdsSubscriber::ros2_to_dds_topic("/imu"), "rt/imu");
    }

    #[test]
    fn ros2_to_dds_topic_no_slash() {
        assert_eq!(DdsSubscriber::ros2_to_dds_topic("imu"), "rt/imu");
    }

    #[test]
    fn ros2_to_dds_topic_already_prefixed() {
        assert_eq!(DdsSubscriber::ros2_to_dds_topic("rt/imu"), "rt/imu");
    }

    #[test]
    fn ros2_to_dds_topic_nested_path() {
        assert_eq!(
            DdsSubscriber::ros2_to_dds_topic("/camera/image_raw"),
            "rt/camera/image_raw"
        );
    }

    #[test]
    fn ros2_to_dds_topic_empty_string() {
        assert_eq!(DdsSubscriber::ros2_to_dds_topic(""), "rt/");
    }

    #[test]
    fn ros2_type_to_dds_type_three_part() {
        assert_eq!(
            DdsSubscriber::ros2_type_to_dds_type("sensor_msgs/msg/Imu"),
            "sensor_msgs::msg::dds_::Imu_"
        );
    }

    #[test]
    fn ros2_type_to_dds_type_geometry_msgs() {
        assert_eq!(
            DdsSubscriber::ros2_type_to_dds_type("geometry_msgs/msg/PointStamped"),
            "geometry_msgs::msg::dds_::PointStamped_"
        );
    }

    #[test]
    fn ros2_type_to_dds_type_non_three_part_passthrough() {
        assert_eq!(
            DdsSubscriber::ros2_type_to_dds_type("custom_type"),
            "custom_type"
        );
    }

    #[test]
    fn ros2_type_to_dds_type_two_part_passthrough() {
        assert_eq!(
            DdsSubscriber::ros2_type_to_dds_type("sensor_msgs/Imu"),
            "sensor_msgs/Imu"
        );
    }

    #[test]
    fn subscriber_new_rejects_empty_topic() {
        let config = DdsSubscriberConfig {
            topic: String::new(),
            ..Default::default()
        };
        assert!(matches!(
            DdsSubscriber::new(config),
            Err(DdsError::InvalidTopicName { .. })
        ));
    }

    #[test]
    fn subscriber_new_accepts_valid_config() {
        let config = DdsSubscriberConfig {
            topic: "/imu".to_string(),
            message_type: "sensor_msgs/msg/Imu".to_string(),
            ..Default::default()
        };
        assert!(DdsSubscriber::new(config).is_ok());
    }

    #[test]
    fn subscriber_config_default_values() {
        let config = DdsSubscriberConfig::default();
        assert_eq!(config.domain_id, 0);
        assert_eq!(config.channel_capacity, 4096);
        assert_eq!(config.history_depth, 10);
        assert!(config.topic.is_empty());
        assert!(config.message_type.is_empty());
        assert!(matches!(config.reliability, QosReliability::BestEffort));
        assert!(matches!(config.durability, QosDurability::Volatile));
    }
}
