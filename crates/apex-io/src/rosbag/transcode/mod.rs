//! Schema-driven conversion of message payloads between ROS1 and ROS2 (CDR).
//!
//! The converter is compiled from the message definition carried by the bag
//! (ROS1 connection header, MCAP schema) or, for ROS2 sqlite bags that store
//! none, from a built-in catalog of the standard interface packages. It
//! therefore handles every message type with a known definition -- including
//! user-defined ones -- without hand-written per-type code, and copies large
//! byte arrays (images, point clouds) without touching their elements.

mod catalog;
pub mod msgdef;
mod plan;

pub use catalog::catalog;
pub use msgdef::{Dialect, Registry, dialect_type_name, normalize_type_name};
pub use plan::Transcoder;

use crate::rosbag::bag::NewConnection;
use crate::rosbag::types::Connection;

/// A ready-to-use conversion for one topic.
#[derive(Debug, Clone)]
pub struct TopicPlan {
    /// Payload converter.
    pub transcoder: Transcoder,
    /// How the topic must be declared in the output bag.
    pub target: NewConnection,
}

/// Definition text for `message_type` in `dialect` from the built-in catalog,
/// or `None` for types outside it.
pub fn catalog_definition(message_type: &str, dialect: Dialect) -> Option<String> {
    let root = normalize_type_name(message_type);
    let cat = catalog().ok()?;
    if !cat.contains(&root) {
        return None;
    }
    cat.render(&root, dialect).ok()
}

/// ROS1 md5sum of a type from the built-in catalog.
pub fn catalog_md5(message_type: &str) -> Option<String> {
    let root = normalize_type_name(message_type);
    let cat = catalog().ok()?;
    cat.contains(&root).then(|| cat.md5(&root).ok()).flatten()
}

/// Payload dialect named by a connection's `serialization_format`.
pub fn dialect_of(serialization_format: &str) -> Option<Dialect> {
    match serialization_format {
        "ros1" => Some(Dialect::Ros1),
        "cdr" | "" => Some(Dialect::Ros2),
        _ => None,
    }
}

/// Build the conversion of `conn` into dialect `to`.
///
/// `Err` carries the human-readable reason the topic cannot be converted
/// (callers log it and skip the topic).
pub fn plan_topic(conn: &Connection, to: Dialect) -> std::result::Result<TopicPlan, String> {
    let from = dialect_of(&conn.serialization_format).ok_or_else(|| {
        format!(
            "unsupported payload encoding '{}'",
            conn.serialization_format
        )
    })?;
    let root = normalize_type_name(&conn.message_type);

    let mut reg = if conn.message_definition.data.trim().is_empty() {
        Registry::default()
    } else {
        Registry::parse(&root, &conn.message_definition.data).map_err(|e| e.to_string())?
    };
    if !reg.contains(&root) || !reg.missing_from(&root).is_empty() {
        let cat = catalog().map_err(|e| format!("built-in catalog unavailable: {e}"))?;
        reg.fill_missing_from(cat);
    }
    if !reg.contains(&root) {
        return Err(format!(
            "no message definition for '{}' (not in the bag, not a standard type)",
            conn.message_type
        ));
    }
    let missing = reg.missing_from(&root);
    if !missing.is_empty() {
        return Err(format!(
            "'{}' depends on undefined type(s): {}",
            conn.message_type,
            missing.join(", ")
        ));
    }

    let transcoder = Transcoder::compile(&reg, &root, from, to).map_err(|e| e.to_string())?;
    let definition = reg.render(&root, to).map_err(|e| e.to_string())?;
    let type_hash = match to {
        Dialect::Ros1 => reg.md5(&root).map_err(|e| e.to_string())?,
        Dialect::Ros2 => String::new(),
    };
    Ok(TopicPlan {
        transcoder,
        target: NewConnection {
            topic: conn.topic.clone(),
            message_type: dialect_type_name(&root, to),
            definition,
            type_hash,
            serialization_format: match to {
                Dialect::Ros1 => "ros1",
                Dialect::Ros2 => "cdr",
            }
            .to_string(),
            qos: conn.offered_qos_profiles.clone(),
        },
    })
}

/// A topic that cannot be converted, and why.
#[derive(Debug, Clone)]
pub struct Unsupported {
    /// Topic name.
    pub topic: String,
    /// Message type as stored in the source bag.
    pub message_type: String,
    /// Human-readable reason.
    pub reason: String,
}

/// Plan every connection in `connections` for dialect `to`.
///
/// Topics that cannot be converted are returned in the second list and are
/// reported with a `tracing::warn!` each, so callers never drop data
/// silently.
pub fn plan_connections(
    connections: &[Connection],
    to: Dialect,
) -> (Vec<(Connection, TopicPlan)>, Vec<Unsupported>) {
    let mut planned = Vec::new();
    let mut unsupported = Vec::new();
    for c in connections {
        match plan_topic(c, to) {
            Ok(plan) => planned.push((c.clone(), plan)),
            Err(reason) => {
                tracing::warn!(
                    topic = %c.topic,
                    message_type = %c.message_type,
                    "skipping topic: {reason}"
                );
                unsupported.push(Unsupported {
                    topic: c.topic.clone(),
                    message_type: c.message_type.clone(),
                    reason,
                });
            }
        }
    }
    (planned, unsupported)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rosbag::types::{MessageDefinition, MessageDefinitionFormat};

    fn conn(topic: &str, ty: &str, fmt: &str, def: &str) -> Connection {
        Connection {
            id: 1,
            topic: topic.into(),
            message_type: ty.into(),
            message_definition: MessageDefinition {
                format: MessageDefinitionFormat::Msg,
                data: def.into(),
            },
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: fmt.into(),
            offered_qos_profiles: Vec::new(),
        }
    }

    #[test]
    fn standard_ros2_type_without_embedded_definition_uses_the_catalog() {
        let c = conn("/imu", "sensor_msgs/msg/Imu", "cdr", "");
        let plan = plan_topic(&c, Dialect::Ros1).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.target.message_type, "sensor_msgs/Imu");
        assert_eq!(plan.target.type_hash, "6a62c6daae103f4ff57a132d6f95cec2");
        assert!(plan.target.definition.starts_with("Header header"));
        assert_eq!(plan.target.serialization_format, "ros1");
    }

    #[test]
    fn user_defined_types_convert_from_their_embedded_definition() {
        let c = conn(
            "/mine",
            "my_pkg/Thing",
            "ros1",
            "Header header\nint16 id\nstring[] tags\n",
        );
        let plan = plan_topic(&c, Dialect::Ros2).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.target.message_type, "my_pkg/msg/Thing");
        assert!(
            plan.target
                .definition
                .contains("builtin_interfaces/Time stamp")
        );
    }

    #[test]
    fn unknown_types_without_a_definition_are_reported_not_dropped_silently() {
        let known = conn("/ok", "std_msgs/String", "ros1", "string data\n");
        let unknown = conn("/odd", "vendor_msgs/Mystery", "cdr", "");
        let weird = conn("/weird", "std_msgs/String", "json", "");
        let (planned, unsupported) = plan_connections(&[known, unknown, weird], Dialect::Ros1);
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].0.topic, "/ok");
        assert_eq!(unsupported.len(), 2);
        assert!(unsupported[0].reason.contains("no message definition"));
        assert!(unsupported[1].reason.contains("payload encoding"));
    }
}
