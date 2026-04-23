use crate::rosbag::types::{QosDurability, QosReliability};
use rustdds::policy::{Durability, History, Reliability};

pub fn to_dds_reliability(r: &QosReliability) -> Reliability {
    match r {
        QosReliability::Reliable => Reliability::Reliable {
            max_blocking_time: rustdds::Duration::from_millis(100),
        },
        _ => Reliability::BestEffort,
    }
}

pub fn to_dds_durability(d: &QosDurability) -> Durability {
    match d {
        QosDurability::TransientLocal => Durability::TransientLocal,
        _ => Durability::Volatile,
    }
}

pub fn to_dds_history(depth: i32) -> History {
    if depth <= 0 {
        History::KeepAll
    } else {
        History::KeepLast { depth }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::rosbag::types::{QosDurability, QosReliability};

    #[test]
    fn reliability_reliable_maps_to_reliable() {
        assert!(matches!(
            to_dds_reliability(&QosReliability::Reliable),
            Reliability::Reliable { .. }
        ));
    }

    #[test]
    fn reliability_best_effort_maps_to_best_effort() {
        assert!(matches!(
            to_dds_reliability(&QosReliability::BestEffort),
            Reliability::BestEffort
        ));
    }

    #[test]
    fn reliability_system_default_maps_to_best_effort() {
        assert!(matches!(
            to_dds_reliability(&QosReliability::SystemDefault),
            Reliability::BestEffort
        ));
    }

    #[test]
    fn reliability_unknown_maps_to_best_effort() {
        assert!(matches!(
            to_dds_reliability(&QosReliability::Unknown),
            Reliability::BestEffort
        ));
    }

    #[test]
    fn durability_transient_local_maps_to_transient_local() {
        assert!(matches!(
            to_dds_durability(&QosDurability::TransientLocal),
            Durability::TransientLocal
        ));
    }

    #[test]
    fn durability_volatile_maps_to_volatile() {
        assert!(matches!(
            to_dds_durability(&QosDurability::Volatile),
            Durability::Volatile
        ));
    }

    #[test]
    fn durability_system_default_maps_to_volatile() {
        assert!(matches!(
            to_dds_durability(&QosDurability::SystemDefault),
            Durability::Volatile
        ));
    }

    #[test]
    fn history_zero_depth_maps_to_keep_all() {
        assert!(matches!(to_dds_history(0), History::KeepAll));
    }

    #[test]
    fn history_negative_depth_maps_to_keep_all() {
        assert!(matches!(to_dds_history(-1), History::KeepAll));
    }

    #[test]
    fn history_positive_depth_maps_to_keep_last() {
        match to_dds_history(5) {
            History::KeepLast { depth } => assert_eq!(depth, 5),
            _ => panic!("expected KeepLast"),
        }
    }

    #[test]
    fn history_depth_one_maps_to_keep_last() {
        assert!(matches!(to_dds_history(1), History::KeepLast { depth: 1 }));
    }
}
