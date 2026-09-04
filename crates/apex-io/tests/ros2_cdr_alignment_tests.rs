//! CDR alignment conformance for the ROS 2 deserializer.
//!
//! # Why the fixture is built by hand
//!
//! OMG CDR measures each primitive's alignment from the start of the
//! **encapsulated body** — after the 4-byte encapsulation header — not from the
//! start of the buffer. The two agree for 1, 2 and 4-byte types, because the
//! header is exactly 4 bytes, and disagree for every 8-byte type that follows a
//! variable-length field. The symptom is a `float64` read 4 bytes off, which
//! decodes as a plausible-looking but absurd number, classically ~1e200.
//!
//! Neither encoder to hand can serve as the oracle:
//!
//! * this crate's own `CdrSerializer` mirrored the same rule, so a round-trip
//!   through it passed while real bags failed;
//! * `rosbags-rs`, the other decoder in this workspace, has the same defect —
//!   it is what apex-vio hand-rolled a `TransformStamped` decoder to work
//!   around.
//!
//! So the buffer below is assembled byte by byte from the specification, with
//! every offset stated. It encodes the standard rather than an implementation.

#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Test diagnostics below print under `--nocapture` only.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::Path;

use apex_io::rosbag::Reader;
use apex_io::rosbag::cdr::CdrDeserializer;
use apex_io::rosbag::messages::{FromCdr, TransformStamped};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const TRANSLATION: [f64; 3] = [1.25, -2.5, 3.75];
const ROTATION: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

/// A `geometry_msgs/TransformStamped` laid out per CDR.
///
/// ```text
/// abs  0..4    encapsulation header (little-endian CDR)
/// abs  4       int32  stamp.sec           body 0,  align 4
/// abs  8       uint32 stamp.nanosec       body 4,  align 4
/// abs 12       uint32 frame_id length = 6 body 8,  align 4
/// abs 16..22   "world\0"
/// abs 24       uint32 child length = 5    body 20, align 4  (padded from 22)
/// abs 28..33   "base\0"
/// abs 36       float64 translation.x      body 32, align 8  (padded from 33)
/// abs 44, 52   translation.y, .z
/// abs 60..92   rotation x, y, z, w
/// ```
///
/// The load-bearing offset is 36. Aligning from the buffer start instead would
/// place `translation.x` at 40 — inside the gap and straddling x and y.
fn transform_stamped_bytes() -> Vec<u8> {
    let mut buf = Vec::with_capacity(92);

    // Encapsulation header: CDR_LE.
    buf.extend_from_slice(&[0x00, 0x01, 0x00, 0x00]);

    buf.extend_from_slice(&100i32.to_le_bytes()); // abs 4  sec
    buf.extend_from_slice(&200u32.to_le_bytes()); // abs 8  nanosec

    buf.extend_from_slice(&6u32.to_le_bytes()); // abs 12 frame_id length
    buf.extend_from_slice(b"world\0"); // abs 16..22
    assert_eq!(buf.len(), 22);

    while (buf.len() - 4) % 4 != 0 {
        buf.push(0); // pad body 18 -> 20
    }
    assert_eq!(buf.len(), 24);
    buf.extend_from_slice(&5u32.to_le_bytes()); // abs 24 child length
    buf.extend_from_slice(b"base\0"); // abs 28..33
    assert_eq!(buf.len(), 33);

    while (buf.len() - 4) % 8 != 0 {
        buf.push(0); // pad body 29 -> 32
    }
    assert_eq!(
        buf.len(),
        36,
        "translation must begin at absolute offset 36"
    );

    for value in TRANSLATION {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    for value in ROTATION {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(buf.len(), 92);
    buf
}

/// Decoding a spec-conformant message must recover the values written.
///
/// This is the whole bug in one assertion: with buffer-relative alignment,
/// `translation.x` is read from offset 40 and comes back as garbage.
#[test]
fn transform_stamped_decodes_with_body_relative_alignment() -> TestResult {
    let bytes = transform_stamped_bytes();
    let mut deserializer = CdrDeserializer::new(&bytes)?;
    let message = TransformStamped::from_cdr(&mut deserializer)?;

    assert_eq!(message.header.stamp.sec, 100);
    assert_eq!(message.header.stamp.nanosec, 200);
    assert_eq!(message.header.frame_id, "world");
    assert_eq!(message.child_frame_id, "base");

    let t = &message.transform.translation;
    for (got, want) in [
        (t.x, TRANSLATION[0]),
        (t.y, TRANSLATION[1]),
        (t.z, TRANSLATION[2]),
    ] {
        assert!(
            (got - want).abs() < 1e-12,
            "translation decoded as {got}, expected {want} — \
             this is the buffer-relative alignment bug"
        );
    }

    let r = &message.transform.rotation;
    assert!((r.w - 1.0).abs() < 1e-12, "rotation.w decoded as {}", r.w);
    Ok(())
}

/// The four-byte types agree under either rule, because the header is itself
/// four bytes. Pinning that keeps a "fix" that shifts them from passing
/// unnoticed.
#[test]
fn four_byte_fields_are_unaffected_by_the_alignment_rule() -> TestResult {
    let bytes = transform_stamped_bytes();
    let mut deserializer = CdrDeserializer::new(&bytes)?;
    let message = TransformStamped::from_cdr(&mut deserializer)?;

    assert_eq!(message.header.stamp.sec, 100);
    assert_eq!(message.header.stamp.nanosec, 200);
    Ok(())
}

/// A float64 immediately after the header needs no padding, so it lands at the
/// same offset either way. Guards the fix against over-correcting.
#[test]
fn a_float64_at_the_start_of_the_body_needs_no_padding() -> TestResult {
    let mut buf = vec![0x00, 0x01, 0x00, 0x00];
    buf.extend_from_slice(&(-4.5f64).to_le_bytes());

    let mut deserializer = CdrDeserializer::new(&buf)?;
    let value = deserializer.read_f64()?;
    assert!((value + 4.5).abs() < 1e-12, "got {value}");
    Ok(())
}

/// A bag written by real ROS 2 tooling, which is the only encoder in reach that
/// does not share the defect under test.
const EUROC_BAG: &str = "/Volumes/External/Workspace/rust/apex-vio/data/V1_01_easy";

/// End-to-end proof on real data.
///
/// EuRoC V1_01 is a 1.5 m-wide Vicon room, so every translation must be a few
/// metres at most. Before the fix this topic decoded to ~1e200 — the classic
/// signature of reading a `float64` from the wrong offset.
#[test]
fn a_real_ros2_bag_decodes_plausible_vicon_translations() -> TestResult {
    if !Path::new(EUROC_BAG).is_dir() {
        eprintln!("Skipping: bag not available at {EUROC_BAG}");
        return Ok(());
    }

    let mut reader = Reader::new(Path::new(EUROC_BAG))?;
    reader.open()?;

    let mut checked = 0usize;
    for message in reader.messages()? {
        let message = message?;
        if message.topic != "/vicon/firefly_sbx/firefly_sbx" {
            continue;
        }
        let mut deserializer = CdrDeserializer::new(&message.data)?;
        let transform = TransformStamped::from_cdr(&mut deserializer)?;
        let t = &transform.transform.translation;

        for (axis, value) in [("x", t.x), ("y", t.y), ("z", t.z)] {
            assert!(
                value.is_finite() && value.abs() < 100.0,
                "translation.{axis} = {value} — a Vicon room is metres across, \
                 so this is the alignment bug, not real motion"
            );
        }

        if checked == 0 {
            // The first sample of V1_01, decoded correctly. Pinned so a future
            // regression that merely keeps the values finite still fails.
            assert!((t.x - 0.7868).abs() < 1e-3, "first x = {}", t.x);
            assert!((t.y - 2.1766).abs() < 1e-3, "first y = {}", t.y);
            assert!((t.z - 1.0620).abs() < 1e-3, "first z = {}", t.z);
        }

        checked += 1;
        if checked >= 200 {
            break;
        }
    }

    assert!(checked > 0, "the Vicon topic should carry messages");
    Ok(())
}
