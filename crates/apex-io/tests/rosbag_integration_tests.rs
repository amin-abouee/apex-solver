//! Integration tests for the `rosbag` module
//!
//! Validates reading ROS2 bag files in SQLite3 and MCAP formats against reference
//! data extracted from the Python rosbags library.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use apex_io::rosbag::types::{CompressionFormat, CompressionMode, StoragePlugin};
use apex_io::rosbag::{Reader, Writer};
use tempfile::tempdir;

use std::collections::HashMap;

const SQLITE3_BAG_PATH: &str = "tests/test_bags/test_bag_sqlite3";

const MCAP_BAG_PATH: &str = "tests/test_bags/test_bag_mcap";

// ---------------------------------------------------------------------------
// Reference data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]

struct ReferenceMessage {
    topic: &'static str,
    msgtype: &'static str,
    raw_data_hex: &'static str,
    timestamp: u64,
}

#[derive(Debug, Clone, PartialEq)]

struct ReferenceTopic {
    topic: &'static str,
    msgtype: &'static str,
}

#[derive(Debug, Clone, PartialEq)]

struct ExpectedBagMetadata {
    message_count: u64,
    topic_count: usize,
    storage_identifier: &'static str,
}

// ---------------------------------------------------------------------------
// Reference data (extracted from Python rosbags library)
// ---------------------------------------------------------------------------

fn get_sqlite3_reference_data() -> (
    ExpectedBagMetadata,
    Vec<ReferenceTopic>,
    Vec<ReferenceMessage>,
) {
    let metadata = ExpectedBagMetadata {
        message_count: 188,
        topic_count: 94,
        storage_identifier: "sqlite3",
    };

    let topics = vec![
        ReferenceTopic {
            topic: "/test/geometry_msgs/accel",
            msgtype: "geometry_msgs/msg/Accel",
        },
        ReferenceTopic {
            topic: "/test/geometry_msgs/accel_stamped",
            msgtype: "geometry_msgs/msg/AccelStamped",
        },
        ReferenceTopic {
            topic: "/test/std_msgs/string",
            msgtype: "std_msgs/msg/String",
        },
        ReferenceTopic {
            topic: "/test/std_msgs/int32",
            msgtype: "std_msgs/msg/Int32",
        },
        ReferenceTopic {
            topic: "/test/sensor_msgs/image",
            msgtype: "sensor_msgs/msg/Image",
        },
        ReferenceTopic {
            topic: "/test/geometry_msgs/point",
            msgtype: "geometry_msgs/msg/Point",
        },
    ];

    let messages = vec![
        ReferenceMessage {
            topic: "/test/geometry_msgs/accel",
            msgtype: "geometry_msgs/msg/Accel",
            raw_data_hex: "",
            timestamp: 0,
        },
        ReferenceMessage {
            topic: "/test/std_msgs/string",
            msgtype: "std_msgs/msg/String",
            raw_data_hex: "",
            timestamp: 0,
        },
    ];

    (metadata, topics, messages)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_bag_metadata(reader: &Reader, expected: &ExpectedBagMetadata) -> Result<(), String> {
    let metadata = reader.metadata().ok_or("Failed to get metadata")?;
    let info = metadata.info();

    if info.storage_identifier != expected.storage_identifier {
        return Err(format!(
            "Storage identifier mismatch: expected {}, got {}",
            expected.storage_identifier, info.storage_identifier
        ));
    }

    if info.message_count != expected.message_count {
        return Err(format!(
            "Message count mismatch: expected {}, got {}",
            expected.message_count, info.message_count
        ));
    }

    let connections = reader.connections();
    if connections.len() != expected.topic_count {
        return Err(format!(
            "Topic count mismatch: expected {}, got {}",
            expected.topic_count,
            connections.len()
        ));
    }

    Ok(())
}

fn validate_topics(reader: &Reader, expected_topics: &[ReferenceTopic]) -> Result<(), String> {
    let connections = reader.connections();
    let topic_map: HashMap<String, _> = connections.iter().map(|c| (c.topic.clone(), c)).collect();

    for expected_topic in expected_topics {
        match topic_map.get(expected_topic.topic) {
            Some(connection) if connection.msgtype() != expected_topic.msgtype => {
                return Err(format!(
                    "Message type mismatch for topic '{}': expected {}, got {}",
                    expected_topic.topic,
                    expected_topic.msgtype,
                    connection.msgtype()
                ));
            }
            None => {
                return Err(format!(
                    "Expected topic '{}' not found",
                    expected_topic.topic
                ));
            }
            _ => {}
        }
    }

    Ok(())
}

fn validate_messages(
    reader: &mut Reader,
    expected_messages: &[ReferenceMessage],
) -> Result<(), String> {
    let mut message_map: HashMap<String, Vec<_>> = HashMap::new();

    for message_result in reader
        .messages()
        .map_err(|e| format!("Failed to get messages: {e}"))?
    {
        let message = message_result.map_err(|e| format!("Failed to read message: {e}"))?;
        message_map
            .entry(message.topic.clone())
            .or_default()
            .push(message);
    }

    for expected_msg in expected_messages {
        match message_map.get(expected_msg.topic) {
            Some(messages) if messages.is_empty() => {
                return Err(format!(
                    "No messages found for topic '{}'",
                    expected_msg.topic
                ));
            }
            None => {
                return Err(format!(
                    "No messages found for topic '{}'",
                    expected_msg.topic
                ));
            }
            Some(messages) => {
                for msg in messages {
                    if msg.data.is_empty() {
                        return Err(format!(
                            "Empty message data for topic '{}'",
                            expected_msg.topic
                        ));
                    }
                    if msg.timestamp == 0 {
                        return Err(format!(
                            "Invalid timestamp for topic '{}'",
                            expected_msg.topic
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — SQLite3
// ---------------------------------------------------------------------------

#[test]
fn test_read_sqlite3_bag() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    assert!(reader.is_open());

    let (expected_metadata, expected_topics, _) = get_sqlite3_reference_data();
    validate_bag_metadata(&reader, &expected_metadata).expect("Metadata validation failed");
    validate_topics(&reader, &expected_topics).expect("Topic validation failed");

    let mut message_count = 0;
    for message_result in reader.messages().expect("Failed to get messages") {
        let _message = message_result.expect("Failed to read message");
        message_count += 1;
    }
    assert_eq!(message_count, 188);
}

#[test]
fn test_sqlite3_message_validation() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    let (_, _, expected_messages) = get_sqlite3_reference_data();
    validate_messages(&mut reader, &expected_messages).expect("Message validation failed");
}

#[test]
fn test_message_filtering_by_topic() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    let test_topic = "/test/std_msgs/string";
    let filtered_connections: Vec<_> = reader
        .connections()
        .iter()
        .filter(|c| c.topic == test_topic)
        .cloned()
        .collect();

    assert_eq!(filtered_connections.len(), 1);

    let mut message_count = 0;
    for message_result in reader
        .messages_filtered(Some(&filtered_connections), None, None)
        .expect("Failed to get filtered messages")
    {
        let message = message_result.expect("Failed to read message");
        assert_eq!(message.topic, test_topic);
        message_count += 1;
    }

    assert_eq!(message_count, 2);
}

#[test]
fn test_message_filtering_by_timestamp() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    let mut all_timestamps: Vec<u64> = reader
        .messages()
        .expect("Failed to get messages")
        .filter_map(|r| r.ok())
        .map(|m| m.timestamp)
        .collect();

    all_timestamps.sort_unstable();
    let min_ts = all_timestamps[0];
    let max_ts = *all_timestamps.last().unwrap();
    let mid_ts = (min_ts + max_ts) / 2;

    let mut count = 0;
    for message_result in reader
        .messages_filtered(None, Some(min_ts), Some(mid_ts))
        .expect("Failed to get filtered messages")
    {
        let message = message_result.expect("Failed to read message");
        assert!(message.timestamp >= min_ts);
        assert!(message.timestamp <= mid_ts);
        count += 1;
    }

    assert!(count > 0);
    assert!(count < all_timestamps.len());
}

#[test]
fn test_specific_message_types() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    let test_types = [
        "std_msgs/msg/String",
        "std_msgs/msg/Int32",
        "std_msgs/msg/Float64",
        "geometry_msgs/msg/Point",
        "geometry_msgs/msg/Pose",
        "sensor_msgs/msg/Image",
    ];

    for msgtype in &test_types {
        let matching: Vec<_> = reader
            .connections()
            .iter()
            .filter(|c| c.msgtype() == *msgtype)
            .cloned()
            .collect();

        assert_eq!(
            matching.len(),
            1,
            "Expected exactly one connection for message type '{msgtype}'"
        );

        let mut count = 0;
        for result in reader
            .messages_filtered(Some(&matching), None, None)
            .expect("Failed to get filtered messages")
        {
            let message = result.expect("Failed to read message");
            assert!(
                !message.data.is_empty(),
                "Empty message data for type '{msgtype}'"
            );
            assert!(
                message.timestamp > 0,
                "Invalid timestamp for type '{msgtype}'"
            );
            count += 1;
        }
        assert_eq!(count, 2, "Expected 2 messages for type '{msgtype}'");
    }
}

#[test]
fn test_comprehensive_message_type_coverage() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    let connections = reader.connections();

    let expected_categories = [
        ("geometry_msgs", 29usize),
        ("nav_msgs", 5),
        ("sensor_msgs", 27),
        ("std_msgs", 30),
        ("stereo_msgs", 1),
        ("tf2_msgs", 2),
    ];

    let mut category_counts: HashMap<String, usize> = HashMap::new();
    for connection in connections {
        if let Some(category) = connection.msgtype().split('/').next() {
            *category_counts.entry(category.to_string()).or_insert(0) += 1;
        }
    }

    for (category, expected_count) in &expected_categories {
        let actual = category_counts.get(*category).copied().unwrap_or(0);
        assert_eq!(
            actual, *expected_count,
            "Expected {expected_count} types in category '{category}', found {actual}"
        );
    }

    let total: usize = category_counts.values().sum();
    assert_eq!(total, 94, "Expected 94 total message types, found {total}");
}

#[test]
fn test_all_messages_readable() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    let mut message_count = 0;
    let mut topics_seen = std::collections::HashSet::new();

    for message_result in reader.messages().expect("Failed to get messages") {
        let message = message_result.expect("Failed to read message");
        topics_seen.insert(message.topic.clone());
        assert!(
            !message.data.is_empty(),
            "Empty data for topic '{}'",
            message.topic
        );
        assert!(
            message.timestamp > 0,
            "Invalid timestamp for topic '{}'",
            message.topic
        );
        message_count += 1;
    }

    assert_eq!(
        message_count, 188,
        "Expected 188 messages, got {message_count}"
    );
    assert_eq!(
        topics_seen.len(),
        94,
        "Expected 94 unique topics, saw {}",
        topics_seen.len()
    );
}

// ---------------------------------------------------------------------------
// Tests — MCAP
// ---------------------------------------------------------------------------

#[test]
fn test_read_mcap_bag() {
    use apex_io::rosbag::Reader;

    let mut reader = Reader::new(MCAP_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");
    assert!(reader.is_open());

    let topics = reader.topics();
    assert!(!topics.is_empty(), "MCAP bag should have topics");

    let mut message_count = 0;
    for message_result in reader.messages().expect("Failed to get messages") {
        let _message = message_result.expect("Failed to read message");
        message_count += 1;
    }
    assert_eq!(message_count, 188);
}

#[test]
fn test_mcap_message_validation() {
    use apex_io::rosbag::Reader;

    let mut reader = Reader::new(MCAP_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    let mut message_count = 0;
    for message_result in reader.messages().expect("Failed to get messages") {
        let message = message_result.expect("Failed to read message");
        assert!(!message.data.is_empty(), "Message data should not be empty");
        assert!(
            !message.topic.is_empty(),
            "Message topic should not be empty"
        );
        message_count += 1;
    }
    assert!(message_count > 0, "Should have read some messages");
    println!("MCAP message validation: {message_count} messages read");
}

// ---------------------------------------------------------------------------
// Tests — cross-format consistency
// ---------------------------------------------------------------------------

#[test]
fn test_bag_format_consistency() {
    use apex_io::rosbag::Reader;

    let mut sqlite_reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create SQLite reader");
    sqlite_reader.open().expect("Failed to open SQLite bag");

    let mut mcap_reader = Reader::new(MCAP_BAG_PATH).expect("Failed to create MCAP reader");
    mcap_reader.open().expect("Failed to open MCAP bag");

    let sqlite_connections = sqlite_reader.connections();
    let mcap_connections = mcap_reader.connections();

    assert_eq!(
        sqlite_connections.len(),
        mcap_connections.len(),
        "Both bags should have the same number of topics"
    );

    let sqlite_topics: HashMap<String, String> = sqlite_connections
        .iter()
        .map(|c| (c.topic.clone(), c.msgtype().to_string()))
        .collect();

    let mcap_topics: HashMap<String, String> = mcap_connections
        .iter()
        .map(|c| (c.topic.clone(), c.msgtype().to_string()))
        .collect();

    for (topic, msgtype) in &sqlite_topics {
        assert!(
            mcap_topics.contains_key(topic),
            "Topic '{topic}' missing from MCAP bag"
        );
        assert_eq!(
            mcap_topics.get(topic).unwrap(),
            msgtype,
            "Message type mismatch for topic '{topic}'"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests — Writer
// ---------------------------------------------------------------------------

fn make_test_connection(
    writer: &mut Writer,
    topic: &str,
    msgtype: &str,
) -> apex_io::rosbag::Connection {
    writer
        .add_connection(
            topic.to_string(),
            msgtype.to_string(),
            None,
            None,
            None,
            None,
        )
        .expect("add_connection failed")
}

#[test]
fn test_writer_roundtrip_sqlite3() {
    let dir = tempdir().expect("tempdir");
    let bag_path = dir.path().join("roundtrip_bag");

    let mut writer =
        Writer::new(&bag_path, None, Some(StoragePlugin::Sqlite3)).expect("Writer::new");
    writer.open().expect("open");

    let conn = make_test_connection(&mut writer, "/test/string", "std_msgs/msg/String");

    let payload = vec![
        0x00u8, 0x01, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o', 0x00,
    ];
    writer.write(&conn, 1_000_000_000, &payload).expect("write");
    writer.write(&conn, 2_000_000_000, &payload).expect("write");
    writer.close().expect("close");

    let mut reader = Reader::new(&bag_path).expect("Reader::new");
    reader.open().expect("open reader");
    assert!(reader.is_open());

    let count = reader
        .messages()
        .expect("messages")
        .filter_map(|r| r.ok())
        .count();
    assert_eq!(count, 2);
}

#[test]
fn test_writer_duplicate_connection_returns_err() {
    let dir = tempdir().expect("tempdir");
    let bag_path = dir.path().join("dup_conn_bag");

    let mut writer =
        Writer::new(&bag_path, None, Some(StoragePlugin::Sqlite3)).expect("Writer::new");
    writer.open().expect("open");

    make_test_connection(&mut writer, "/imu", "sensor_msgs/msg/Imu");
    let result = writer.add_connection(
        "/imu".to_string(),
        "sensor_msgs/msg/Imu".to_string(),
        None,
        None,
        None,
        None,
    );
    assert!(result.is_err());
}

#[test]
fn test_writer_batch_write_produces_same_output() {
    let dir = tempdir().expect("tempdir");
    let bag_seq = dir.path().join("bag_seq");
    let bag_batch = dir.path().join("bag_batch");

    let payload = vec![0x00u8, 0x01, 0x00, 0x00, 0x01];
    let n = 50usize;

    // Sequential write
    {
        let mut w = Writer::new(&bag_seq, None, Some(StoragePlugin::Sqlite3)).unwrap();
        w.open().unwrap();
        let conn = make_test_connection(&mut w, "/data", "std_msgs/msg/String");
        for i in 0..n {
            w.write(&conn, i as u64 * 1000, &payload).unwrap();
        }
        w.close().unwrap();
    }

    // Batch write
    {
        let mut w = Writer::new(&bag_batch, None, Some(StoragePlugin::Sqlite3)).unwrap();
        w.open().unwrap();
        let conn = make_test_connection(&mut w, "/data", "std_msgs/msg/String");
        let msgs: Vec<(apex_io::rosbag::Connection, u64, Vec<u8>)> = (0..n)
            .map(|i| (conn.clone(), i as u64 * 1000, payload.clone()))
            .collect();
        w.write_raw_messages_batch(&msgs).unwrap();
        w.close().unwrap();
    }

    let count_seq = {
        let mut r = Reader::new(&bag_seq).unwrap();
        r.open().unwrap();
        r.messages().unwrap().filter_map(|m| m.ok()).count()
    };
    let count_batch = {
        let mut r = Reader::new(&bag_batch).unwrap();
        r.open().unwrap();
        r.messages().unwrap().filter_map(|m| m.ok()).count()
    };

    assert_eq!(count_seq, n);
    assert_eq!(count_batch, n);
}

#[test]
fn test_messages_filtered_single_topic() {
    let dir = tempdir().expect("tempdir");
    let bag_path = dir.path().join("filtered_bag");

    let mut writer = Writer::new(&bag_path, None, Some(StoragePlugin::Sqlite3)).unwrap();
    writer.open().unwrap();

    let conn_a = make_test_connection(&mut writer, "/topic_a", "std_msgs/msg/String");
    let conn_b = make_test_connection(&mut writer, "/topic_b", "std_msgs/msg/String");

    let payload = vec![0x00u8, 0x01, 0x00, 0x00, 0x01];
    for i in 0..3u64 {
        writer.write(&conn_a, i * 1000, &payload).unwrap();
        writer.write(&conn_b, i * 1000 + 500, &payload).unwrap();
    }
    writer.close().unwrap();

    let mut reader = Reader::new(&bag_path).unwrap();
    reader.open().unwrap();

    let a_conns: Vec<_> = reader
        .connections()
        .iter()
        .filter(|c| c.topic == "/topic_a")
        .cloned()
        .collect();

    let count_a = reader
        .messages_filtered(Some(&a_conns), None, None)
        .unwrap()
        .filter_map(|m| m.ok())
        .inspect(|m| assert_eq!(m.topic, "/topic_a"))
        .count();

    assert_eq!(count_a, 3);
}

#[test]
fn test_raw_messages_batch_returns_correct_count() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    let batch = reader
        .read_raw_messages_batch(None, None, None)
        .expect("Failed to read batch");

    assert_eq!(batch.len(), 188);
    for msg in &batch {
        assert!(!msg.raw_data.is_empty());
        assert!(msg.timestamp > 0);
    }
}

#[test]
fn test_raw_messages_iter_matches_batch() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    let iter_count = reader
        .raw_messages()
        .expect("raw_messages")
        .filter_map(|r| r.ok())
        .count();

    let batch_count = reader
        .read_raw_messages_batch(None, None, None)
        .expect("batch")
        .len();

    assert_eq!(iter_count, batch_count);
}

#[test]
fn test_reader_timestamps_are_ordered() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    let timestamps: Vec<u64> = reader
        .messages()
        .expect("messages")
        .filter_map(|r| r.ok())
        .map(|m| m.timestamp)
        .collect();

    let mut sorted = timestamps.clone();
    sorted.sort_unstable();
    assert_eq!(timestamps, sorted, "Messages should be in timestamp order");
}

#[test]
fn test_reader_metadata_duration_and_times() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("Failed to create reader");
    reader.open().expect("Failed to open bag");

    assert!(reader.duration() > 0);
    assert!(reader.start_time() > 0);
    assert!(reader.end_time() >= reader.start_time());
    assert_eq!(reader.message_count(), 188);
}

#[test]
fn test_writer_set_compression_after_open_fails() {
    let dir = tempdir().expect("tempdir");
    let bag_path = dir.path().join("comp_test");
    let mut writer = Writer::new(&bag_path, None, Some(StoragePlugin::Sqlite3)).unwrap();
    writer.open().unwrap();
    let result = writer.set_compression(CompressionMode::File, CompressionFormat::Zstd);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Tests — read_bag_metadata_fast
// ---------------------------------------------------------------------------

#[test]
fn test_read_bag_metadata_fast_sqlite3() {
    let meta =
        apex_io::rosbag::read_bag_metadata_fast(SQLITE3_BAG_PATH).expect("should read metadata");
    assert_eq!(meta.message_count(), 188);
    assert!(meta.duration() > 0);
}

#[test]
fn test_read_bag_metadata_fast_mcap() {
    let meta =
        apex_io::rosbag::read_bag_metadata_fast(MCAP_BAG_PATH).expect("should read metadata");
    assert_eq!(meta.message_count(), 188);
}

#[test]
fn test_read_bag_metadata_fast_missing_returns_err() {
    let result = apex_io::rosbag::read_bag_metadata_fast("/nonexistent/path/to/bag");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Tests — MCAP storage reader additional coverage
// ---------------------------------------------------------------------------

#[test]
fn test_mcap_messages_filtered_by_connection() {
    use apex_io::rosbag::Reader;
    let mut reader = Reader::new(MCAP_BAG_PATH).expect("reader");
    reader.open().expect("open");

    let conns: Vec<_> = reader
        .connections()
        .iter()
        .filter(|c| c.topic == "/test/std_msgs/string")
        .cloned()
        .collect();

    assert_eq!(conns.len(), 1);
    let count = reader
        .messages_filtered(Some(&conns), None, None)
        .expect("filtered messages")
        .filter_map(|r| r.ok())
        .count();
    assert_eq!(count, 2);
}

#[test]
fn test_mcap_messages_filtered_by_time() {
    use apex_io::rosbag::Reader;
    let mut reader = Reader::new(MCAP_BAG_PATH).expect("reader");
    reader.open().expect("open");

    let start = reader.start_time();
    let mid = start + reader.duration() / 2;

    let count_half = reader
        .messages_filtered(None, Some(start), Some(mid))
        .expect("filtered")
        .filter_map(|r| r.ok())
        .count();

    assert!(count_half > 0);
    assert!(count_half < 188);
}

#[test]
fn test_mcap_raw_messages_filtered_by_connection() {
    use apex_io::rosbag::Reader;
    let mut reader = Reader::new(MCAP_BAG_PATH).expect("reader");
    reader.open().expect("open");

    let conns: Vec<_> = reader
        .connections()
        .iter()
        .filter(|c| c.topic == "/test/std_msgs/string")
        .cloned()
        .collect();

    let count = reader
        .raw_messages_filtered(Some(&conns), None, None)
        .expect("raw filtered")
        .filter_map(|r| r.ok())
        .count();
    assert_eq!(count, 2);
}

#[test]
fn test_mcap_raw_messages_batch_with_filter() {
    use apex_io::rosbag::Reader;
    let mut reader = Reader::new(MCAP_BAG_PATH).expect("reader");
    reader.open().expect("open");

    let conns: Vec<_> = reader
        .connections()
        .iter()
        .filter(|c| c.topic == "/test/std_msgs/string")
        .cloned()
        .collect();

    let batch = reader
        .read_raw_messages_batch(Some(&conns), None, None)
        .expect("batch");
    assert_eq!(batch.len(), 2);
}

#[test]
fn test_reader_raw_messages_filtered_by_time() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("reader");
    reader.open().expect("open");

    let start = reader.start_time();
    let mid = start + reader.duration() / 2;

    let batch = reader
        .read_raw_messages_batch(None, Some(start), Some(mid))
        .expect("batch");
    assert!(!batch.is_empty());
    assert!(batch.len() < 188);
}

#[test]
fn test_reader_close_and_reopen() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("reader");
    reader.open().expect("open");
    assert!(reader.is_open());
    reader.close().expect("close");
    assert!(!reader.is_open());
    reader.open().expect("reopen");
    assert!(reader.is_open());
}

#[test]
fn test_writer_mcap_roundtrip() {
    let dir = tempdir().expect("tempdir");
    let bag_path = dir.path().join("mcap_roundtrip");

    let mut writer = Writer::new(&bag_path, None, Some(StoragePlugin::Mcap)).expect("writer");
    writer.open().expect("open");
    let conn = make_test_connection(&mut writer, "/test/mcap", "std_msgs/msg/String");
    let payload = vec![0x00u8, 0x01, 0x00, 0x00, 0x01];
    writer.write(&conn, 1_000_000, &payload).expect("write");
    writer.close().expect("close");

    // Bag directory and metadata should exist
    assert!(bag_path.join("metadata.yaml").exists());
}

#[test]
fn test_mcap_raw_messages_iteration() {
    let mut reader = Reader::new(MCAP_BAG_PATH).expect("reader");
    reader.open().expect("open");
    let count = reader
        .raw_messages()
        .expect("raw_messages")
        .filter_map(|r| r.ok())
        .count();
    assert_eq!(count, 188);
}

#[test]
fn test_mcap_read_raw_messages_batch() {
    let mut reader = Reader::new(MCAP_BAG_PATH).expect("reader");
    reader.open().expect("open");
    let batch = reader
        .read_raw_messages_batch(None, None, None)
        .expect("batch");
    assert_eq!(batch.len(), 188);
    for msg in &batch {
        assert!(!msg.connection.topic.is_empty());
        assert!(msg.timestamp > 0);
    }
}

#[test]
fn test_mcap_raw_messages_timestamps_ordered() {
    let mut reader = Reader::new(MCAP_BAG_PATH).expect("reader");
    reader.open().expect("open");
    let msgs: Vec<_> = reader
        .raw_messages()
        .expect("raw_messages")
        .filter_map(|r| r.ok())
        .collect();
    let timestamps: Vec<u64> = msgs.iter().map(|m| m.timestamp).collect();
    for w in timestamps.windows(2) {
        assert!(w[0] <= w[1], "timestamps should be non-decreasing");
    }
}

#[test]
fn test_mcap_raw_messages_filtered_by_time_range() {
    let mut reader = Reader::new(MCAP_BAG_PATH).expect("reader");
    reader.open().expect("open");
    let start = reader.start_time();
    let mid = start + reader.duration() / 2;
    let half = reader
        .raw_messages_filtered(None, Some(start), Some(mid))
        .expect("filtered")
        .filter_map(|r| r.ok())
        .count();
    assert!(half > 0 && half < 188);
}

#[test]
fn test_reader_connections_accessor() {
    let mut reader = Reader::new(SQLITE3_BAG_PATH).expect("reader");
    reader.open().expect("open");
    let conns = reader.connections();
    assert!(!conns.is_empty());
    assert_eq!(conns.len(), 94);
}

#[test]
fn test_reader_metadata_accessor() {
    let reader = Reader::new(SQLITE3_BAG_PATH).expect("reader");
    let meta = reader.metadata();
    assert!(meta.is_some());
    assert!(meta.unwrap().message_count() > 0);
}

#[test]
fn test_reader_read_raw_messages_batch_mcap() {
    let mut reader = Reader::new(MCAP_BAG_PATH).expect("reader");
    reader.open().expect("open");
    let conns: Vec<_> = reader
        .connections()
        .iter()
        .filter(|c| c.topic == "/test/std_msgs/string")
        .cloned()
        .collect();
    let batch = reader
        .read_raw_messages_batch(Some(&conns), None, None)
        .expect("batch");
    assert_eq!(batch.len(), 2);
}
