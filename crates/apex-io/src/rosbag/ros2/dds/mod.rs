//! Async DDS subscriber for live ROS2 topic listening.
//!
//! Requires the `dds` feature flag. Uses `rustdds` for DDS middleware and
//! `tokio` for the async runtime.
//!
//! ## ROS2 topic naming
//! ROS2 DDS topics use a `rt/` prefix on the wire. `/imu` becomes `rt/imu`.
//! [`DdsSubscriber`] handles this mapping automatically.
//!
//! ## Quick start
//! ```no_run
//! use apex_io::dds::{DdsSubscriber, DdsSubscriberConfig};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = DdsSubscriberConfig {
//!     topic: "/imu".to_string(),
//!     message_type: "sensor_msgs/msg/Imu".to_string(),
//!     ..Default::default()
//! };
//!
//! let mut rx = DdsSubscriber::new(config)?.listen()?;
//! while let Some(msg) = rx.recv().await {
//!     println!("Received {} bytes on {} at {}", msg.raw_data.len(), msg.connection.topic, msg.timestamp);
//! }
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod listener;
pub mod qos_mapping;
pub mod raw_bytes;
pub mod subscriber;

pub use error::DdsError;
pub use listener::{DdsListener, ReceivedMessage};
pub use subscriber::{DdsSubscriber, DdsSubscriberConfig};
