//! A synthetic VRS file: one small, compressible IMU-like stream (200 Hz) and
//! one large, incompressible camera-like stream (10 Hz) — the shape of a real
//! Aria recording without depending on one.

use super::Rng;
use apex_io::vrs::writer::stream_tags;
use apex_io::vrs::{DataLayout, LayoutWriter, RecordType, StreamId, VrsWriteOptions, VrsWriter};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub struct VrsFixture {
    pub path: PathBuf,
    pub imu_stream: StreamId,
    pub image_stream: StreamId,
}

const IMU_LAYOUT_JSON: &str = r#"{"data_layout":[
    {"name":"seq","type":"DataPieceValue<uint32_t>","offset":0},
    {"name":"temperature","type":"DataPieceValue<float>","offset":4},
    {"name":"accel","type":"DataPieceArray<float>","size":3,"offset":8},
    {"name":"gyro","type":"DataPieceArray<float>","size":3,"offset":20},
    {"name":"label","type":"DataPieceString","index":0}]}"#;

const IMAGE_LAYOUT_JSON: &str = r#"{"data_layout":[
    {"name":"seq","type":"DataPieceValue<uint64_t>","offset":0},
    {"name":"frame_number","type":"DataPieceValue<uint32_t>","offset":8},
    {"name":"exposure","type":"DataPieceValue<double>","offset":12},
    {"name":"gain","type":"DataPieceValue<double>","offset":20},
    {"name":"capture_ts","type":"DataPieceValue<double>","offset":28},
    {"name":"label","type":"DataPieceString","index":0}]}"#;

const IMU_CUSTOM_BYTES: usize = 168;
const IMAGE_CUSTOM_BYTES: usize = 30_000;
const CAMERA_HZ: u64 = 10;
const IMU_HZ: u64 = 200;
const DURATION_S: u64 = 20;

pub fn write_vrs(dir: &Path) -> VrsFixture {
    let imu_layout = DataLayout::from_json(IMU_LAYOUT_JSON).expect("imu layout");
    let image_layout = DataLayout::from_json(IMAGE_LAYOUT_JSON).expect("image layout");
    let imu_stream = StreamId::new(500, 1);
    let image_stream = StreamId::new(501, 1);
    let path = dir.join("synth.vrs");
    let mut w = VrsWriter::create(&path, VrsWriteOptions::default()).expect("create vrs");
    w.add_stream(imu_stream, stream_tags("Simulated Imu", BTreeMap::new()))
        .expect("add imu stream");
    w.add_stream(image_stream, stream_tags("Simulated Camera", BTreeMap::new()))
        .expect("add image stream");
    w.define_record_format(imu_stream, RecordType::Data, 1, "data_layout+custom", &[&imu_layout])
        .expect("imu format");
    w.define_record_format(
        image_stream,
        RecordType::Data,
        1,
        "data_layout+custom",
        &[&image_layout],
    )
    .expect("image format");

    for i in 0..(DURATION_S * IMU_HZ) as u32 {
        let mut lw = LayoutWriter::new(&imu_layout);
        lw.set_u32("seq", i).expect("seq");
        lw.set_f32("temperature", 30.0 + (i % 100) as f32 * 0.01)
            .expect("temp");
        lw.set_f32_array("accel", &[9.81, 0.0, -0.1]).expect("accel");
        lw.set_f32_array("gyro", &[0.01, -0.02, 0.03]).expect("gyro");
        lw.set_string("label", &format!("imu#{i}")).expect("label");
        let mut payload = lw.finish();
        payload.resize(payload.len() + IMU_CUSTOM_BYTES, 0);
        w.write_record(imu_stream, RecordType::Data, 1, f64::from(i) * 0.005, &payload)
            .expect("imu record");
    }

    let mut rng = Rng::new(0x0123_4567_89AB_CDEF);
    let mut jpeg = vec![0u8; IMAGE_CUSTOM_BYTES];
    for i in 0..(DURATION_S * CAMERA_HZ) as u32 {
        rng.fill_bytes(&mut jpeg);
        let mut lw = LayoutWriter::new(&image_layout);
        lw.set_u64("seq", u64::from(i)).expect("seq");
        lw.set_u32("frame_number", i).expect("frame");
        lw.set_f64("exposure", 0.002).expect("exposure");
        lw.set_f64("gain", 1.5).expect("gain");
        lw.set_f64("capture_ts", f64::from(i) * 0.05 + 0.0025).expect("ts");
        lw.set_string("label", &format!("cam#{i}")).expect("label");
        let mut payload = lw.finish();
        payload.extend_from_slice(&jpeg);
        w.write_record(
            image_stream,
            RecordType::Data,
            1,
            f64::from(i) * 0.05 + 0.0025,
            &payload,
        )
        .expect("image record");
    }
    w.finish().expect("finish vrs");
    VrsFixture {
        path,
        imu_stream,
        image_stream,
    }
}
