# DDS Live Subscription

> **Feature-gated:** everything on this page requires `--features dds` (which
> pulls in `rustdds`, `tokio`, and `futures`). Re-exported at the crate root as
> `apex_io::dds`.

Instead of reading a recorded bag, `rosbag::ros2::dds` subscribes to a **live**
ROS2 stream over DDS/RTPS and delivers raw CDR payloads — the same `RawMessage`
bytes a bag stores. That makes it a drop-in source for recording bags or for
online processing. Two entry points:

- [`DdsSubscriber`](#ddssubscriber-single-topic) — one topic → an async channel.
- [`DdsListener`](#ddslistener-multi-topic) — many topics → per-topic callbacks.

## `DdsSubscriber` (single topic)

```rust
pub struct DdsSubscriberConfig {
    pub topic: String,
    pub message_type: String,
    pub reliability: QosReliability,   // default BestEffort
    pub durability: QosDurability,     // default Volatile
    pub history_depth: i32,            // default 10
    pub domain_id: u16,                // default 0
    pub channel_capacity: usize,       // default 4096
}

pub struct DdsSubscriber;
impl DdsSubscriber {
    pub fn new(config: DdsSubscriberConfig) -> Result<Self>;
    pub fn listen(self) -> Result<mpsc::Receiver<RawMessage>>;
    pub fn ros2_to_dds_topic(ros_topic: &str) -> String;
    pub fn ros2_type_to_dds_type(ros2_type: &str) -> String;
}
```

`listen()` spawns the DDS reader and returns a Tokio `mpsc::Receiver<RawMessage>`;
each received sample is delivered as raw CDR bytes with its timestamp.

The two static helpers implement the ROS2 ↔ DDS name mangling
(e.g. `/imu` → `rt/imu`, `sensor_msgs/msg/Imu` → the DDS type name).

## `DdsListener` (multi-topic)

```rust
pub struct DdsListener;
impl DdsListener {
    pub fn new(domain_id: u16) -> Self;
    pub fn subscribe<F>(self, topic, message_type, callback: F) -> Self
        where F: FnMut(ReceivedMessage) + Send + 'static;
    // … run()/spin to process callbacks indefinitely …
}
```

Builder-style: chain `.subscribe(...)` per topic, then run. Each callback
receives a **`ReceivedMessage`**:

```rust
pub struct ReceivedMessage {
    pub topic: String,
    pub message_type: String,
    pub msg_timestamp_s: Option<f64>,   // from Header.stamp, if present
    pub recv_timestamp_s: f64,          // wall-clock arrival
    pub bytes: usize,
    pub raw_data: Vec<u8>,              // full CDR payload
}
```

This is what the [`dds_multi_listener`](../utils/cli.md#dds_multi_listener) CLI is
built on.

<a id="qos-mapping"></a>
## QoS mapping

`rosbag::ros2::dds::qos_mapping` converts the crate's [QoS types](./overview.md#quality-of-service-qos)
into `rustdds` policies:

| Function | Maps |
|---|---|
| `to_dds_reliability(&QosReliability) -> Reliability` | Reliable / BestEffort |
| `to_dds_durability(&QosDurability) -> Durability` | Volatile / TransientLocal |
| `to_dds_history(depth: i32) -> History` | KeepLast(depth) |

## Raw-bytes decoding

`RawBytes`, `RawBytesDecoder`, and `RawBytesAdapter` are the `rustdds`
deserializer glue that hands the raw CDR sample through unchanged (no typed
decode), so payloads arrive exactly as they went on the wire.

## Errors

DDS-specific failures use `DdsError` (participant/reader creation, topic
mismatch, …); the public channel still yields `RawMessage` values.

## Example

```rust,ignore
use apex_io::dds::{DdsSubscriber, DdsSubscriberConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DdsSubscriberConfig {
        topic: "/imu".into(),
        message_type: "sensor_msgs/msg/Imu".into(),
        domain_id: 0,
        ..Default::default()
    };
    let mut rx = DdsSubscriber::new(config)?.listen()?;
    while let Some(raw) = rx.recv().await {
        println!("{} bytes on {}", raw.raw_data.len(), raw.topic);
    }
    Ok(())
}
```

## References

- OMG *Data Distribution Service (DDS)* and *DDS-RTPS* specifications.
- `rustdds` crate — https://crates.io/crates/rustdds
