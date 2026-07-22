# Message Types

`apex-io` ships strongly-typed Rust structs for the common ROS message types,
with CDR (ROS2) and ROS1-wire (de)serialization. These are what a
[`Message`](./overview.md#connections-and-messages) decodes into, and what you
construct when writing a bag from scratch. Types live in
`rosbag::ros2::messages` and `rosbag::ros1::messages`.

## ROS2 messages (`rosbag::ros2::messages`)

| Category | Types |
|---|---|
| **Primitives / builtin** | `Time`, `Duration`, `Header`, `StdString` |
| **Geometry** | `Vector3`, `Point`, `Point32`, `Quaternion`, `Pose`, `PoseWithCovariance`, `PoseWithCovarianceStamped`, `PoseStamped`, `PointStamped`, `Transform`, `TransformStamped`, `Twist`, `TwistWithCovariance` |
| **Navigation** | `Odometry`, `Path`, `NavSatStatus`, `NavSatFix` |
| **Sensors** | `Imu`, `Image`, `PointField`, `PointCloud2` |
| **Misc** | `ColorRGBA` |

These mirror the standard `std_msgs`, `geometry_msgs`, `nav_msgs`, and
`sensor_msgs` definitions. Each derives (de)serialization so it round-trips
through the [CDR codec](./ros2.md#cdr-de-serialization).

## ROS1 messages (`rosbag::ros1::messages`)

| Category | Types |
|---|---|
| **Common** | `Header` |
| **Geometry** | `Vector3`, `Point`, `Quaternion`, `Pose`, `PoseStamped` |
| **Sensors** | `Imu`, `Image`, `CompressedImage` |
| **Misc** | `StringMsg` |

ROS1 types are decoded/encoded with the little-endian, length-prefixed ROS1 wire
format via [`Ros1Deserializer` / `Ros1Serializer`](./ros1.md#wire-format--low-level-codec).

## Typical structure

The types compose exactly like their ROS counterparts, e.g.:

```rust
// sensor_msgs/msg/Imu (ROS2)
pub struct Imu {
    pub header: Header,
    pub orientation: Quaternion,
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: Vector3,
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: Vector3,
    pub linear_acceleration_covariance: [f64; 9],
}
```

## Using them

- **Reading:** iterate `reader.messages()` and match on the topic's
  `message_type`, then decode the payload into the matching struct via the CDR /
  ROS1 codec.
- **Writing from scratch:** build a struct, serialize it, and pass the bytes to
  `writer.write(&connection, timestamp_ns, &bytes)`. The
  [`write_dummy_bag`](../utils/cli.md#write_dummy_bag) CLI demonstrates this for
  every supported ROS2 type.

## Extending

To support a message type not listed here, add a struct in the relevant
`messages` module and implement its CDR (ROS2) or ROS1-wire (de)serialization
using the primitives in [`cdr`](./ros2.md#cdr-de-serialization) or
[`Ros1Serializer`/`Ros1Deserializer`](./ros1.md). The reader/writer plumbing is
type-agnostic — it moves bytes keyed by a `Connection`, so no reader/writer
changes are needed.
