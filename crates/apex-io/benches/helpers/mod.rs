//! Deterministic synthetic fixtures shared by the apex-io benches.
//!
//! Payloads come from a fixed xorshift PRNG so runs are reproducible. Camera
//! and lidar bytes are incompressible (like real JPEG frames and point
//! clouds), the IMU custom block is zeros (like real compressible IMU
//! records). Everything is written through the crate's own public APIs, so
//! the benches measure exactly what users get.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Writable root for bench fixtures: `<target>/apex-io-bench`. Living under
/// `target/` keeps the (multi-GB) fixtures off the system volume, next to
/// everything else `cargo clean` removes.
pub fn bench_root() -> PathBuf {
    let exe = std::env::current_exe().expect("current exe");
    let target = exe.ancestors().nth(3).expect("bench exe below target/");
    let dir = target.join("apex-io-bench");
    std::fs::create_dir_all(&dir).expect("create bench dir");
    dir
}

/// Wipe and recreate [`bench_root`]; call once from fixture init so a
/// crashed run cannot leak gigabytes of stale artifacts.
pub fn reset_bench_dir() {
    let _ = std::fs::remove_dir_all(bench_root());
    std::fs::create_dir_all(bench_root()).expect("recreate bench dir");
}

/// Per-iteration scratch directory, removed untimed when dropped (criterion
/// drops `iter_batched` outputs outside the measurement).
pub struct Scratch(pub PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Fresh subdirectory of [`bench_root`] for one bench iteration.
pub fn scratch_dir(label: &str) -> Scratch {
    static N: AtomicUsize = AtomicUsize::new(0);
    let p = bench_root().join(format!("{label}-{}", N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&p).expect("create scratch dir");
    Scratch(p)
}

/// Path from an environment variable, if it points at an existing file.
pub fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| p.is_file())
}

/// xorshift64* — tiny, deterministic, good enough to make incompressible data.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn fill_bytes(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let v = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
    }
}

// Each bench crate uses only one half of the fixtures.
#[cfg(feature = "rosbag")]
#[allow(dead_code)]
mod bag;
#[cfg(feature = "rosbag")]
#[allow(unused_imports)]
pub use bag::{Message, Scenario, TopicSpec, ros2_type, write_bag};

#[cfg(feature = "vrs")]
#[allow(dead_code)]
mod vrs;
#[cfg(feature = "vrs")]
#[allow(unused_imports)]
pub use vrs::{VrsFixture, write_vrs};
