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
}
