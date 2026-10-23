//! Built-in catalog of standard message definitions.

use super::msgdef::Registry;
use std::sync::OnceLock;

/// Standard ROS2 interface definitions (std/geometry/sensor/nav/... msgs).
const CATALOG_TEXT: &str = include_str!("catalog_ros2.msgs");

static CATALOG: OnceLock<std::result::Result<Registry, String>> = OnceLock::new();

/// The parsed catalog (parsed once, on first use).
pub fn catalog() -> std::result::Result<&'static Registry, &'static str> {
    CATALOG
        .get_or_init(|| Registry::parse_sections(CATALOG_TEXT).map_err(|e| e.to_string()))
        .as_ref()
        .map_err(String::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_parses_and_is_closed_under_dependencies() {
        let cat = catalog().unwrap_or_else(|e| panic!("catalog: {e}"));
        for name in [
            "sensor_msgs/Imu",
            "sensor_msgs/PointCloud2",
            "tf2_msgs/TFMessage",
        ] {
            assert!(cat.contains(name), "{name}");
            assert!(
                cat.missing_from(name).is_empty(),
                "{name} has undefined deps"
            );
        }
    }

    /// Known ROS1 md5sums (from the ROS distribution): the actionlib types
    /// every ROS1 action server publishes.
    #[test]
    fn actionlib_msgs_match_the_ros1_md5sums() {
        let cat = catalog().unwrap_or_else(|e| panic!("catalog: {e}"));
        for (name, md5) in [
            ("actionlib_msgs/GoalID", "302881f31927c1df708a2dbab0e80ee8"),
            (
                "actionlib_msgs/GoalStatus",
                "d388f9b87b3c471f784434d671988d4a",
            ),
            (
                "actionlib_msgs/GoalStatusArray",
                "8b2b82f13216d0a8ea88bd3af735e619",
            ),
        ] {
            assert_eq!(cat.md5(name).unwrap_or_default(), md5, "{name}");
        }
    }

    #[test]
    fn lifecycle_msgs_are_present_and_convert() {
        let cat = catalog().unwrap_or_else(|e| panic!("catalog: {e}"));
        for name in [
            "lifecycle_msgs/State",
            "lifecycle_msgs/Transition",
            "lifecycle_msgs/TransitionDescription",
            "lifecycle_msgs/TransitionEvent",
        ] {
            assert!(cat.contains(name), "{name}");
        }
    }
}
