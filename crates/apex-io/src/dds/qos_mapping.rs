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
