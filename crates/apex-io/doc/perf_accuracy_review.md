# apex-io ROS1/ROS2/VRS: speed + accuracy review (verified)

Status notes for the optimisation pass on `crates/apex-io` (branch
`feature/vrs-reader-writer`). Every claim in the first draft of this review
was re-checked against the code and re-measured; this version records what
held up, what did not, and what was done. Numbers come from the criterion
benches in `crates/apex-io/benches/` (synthetic VIO-like fixture: 2 × 10 Hz
320×240 cameras, 200 Hz IMU, 5 Hz 10k-point lidar, 20 s, ~48 MB per encoding)
and from the real datasets under `/Volumes/External/datasets`.

```bash
cargo bench -p apex-io --features rosbag,vrs --bench rosbag_bench
APEX_BENCH_VRS=<file.vrs> cargo bench -p apex-io --features vrs --bench vrs_bench -- real_vrs
```

---

## 1. Verdict on the draft's claims

| Claim | Measured (baseline `f16e908` → now) | Verdict |
|---|---|---|
| VRS: reusable zstd contexts, full cursor −36 %, IMU −30 % | −21…−36 %, −28…−29 % (varies with machine load) | **confirmed** |
| VRS: raw passthrough copy −42 % | −48 % on a quiet run; trim −32 % | **confirmed** |
| MCAP drain/filter memoisation −9 / −13 % | −15…−19 % full / two-topic | **confirmed** |
| sqlite: one `GROUP BY`, mmap, k-way merge, −35 % open / −44 % scan | −38 % open, −36 % full, −40 % window | **confirmed** |
| §6 "ROS1 is per-message-bound, parity with MCAP; 1 MiB `BufReader` helps" | ROS1 open was already 2.8 ms (MCAP: 0.02 ms) and the 1 MiB buffer made it **+91 % / read +75 %** | **wrong — regression, fixed (§2.1)** |
| §6 "`IndexData` `ver` is pinned to 0" | the bag 2.0 spec defines it as **1**; our writer emits 1 | **wrong — the check would reject every valid bag (§2.1)** |
| §8 sqlite-only MESSAGE decode via `decode_all` | works, but fresh zstd context per message; MCAP-dir bags (also writable by us) stayed broken | **reworked (§2.2)** |
| §9.2 catalog lacks `lifecycle_msgs`, `actionlib_msgs` | true | **confirmed, added (§2.5)** |
| §9.2 `std_srvs` service-event types missing | srv events carry their definition in the bag's schema; ROS1 has no equivalent | **not needed** |
| §9.3.1 VRS tags are typed (int/double/blob) | Meta's `DescriptionRecord.cpp`: tags are plain `map<string,string>` | **wrong — nothing to implement** |
| §9.3.2 parse `RecordType::Tags` records | Meta: "Tags records are internal to VRS and not exposed in the index" | **moot** |
| §9.3.3 `previous_record_size` validation | real gap, cheap sequentially | **done (§2.6)** |
| §9.3.4 285-1 clock map unused | real gap | **done, opt-in (§2.6)** |
| §9.4 pull iterator + params | real gap for the VIO use case | **done (§2.7)** |

## 2. What changed, and the evidence

### 2.1 ROS1 reader: lazy index, correct validation
`Ros1Reader::open` used to walk every chunk and decode every `IndexData`
record, building a global per-message index it often never needed. It now
reads only the bag header and the tail (connection + chunk-info records, which
already carry per-connection counts and time ranges), cross-checks the header's
`conn_count`/`chunk_count`, and builds the per-message index **per query**:
only chunks overlapping the time window and holding a selected connection are
visited, only the selected connections' `IndexData` records are decoded, and
their entry counts are verified against the chunk infos (a tampered count
fails loudly — tested). `IndexData` `ver` must be 1 (spec).

| ROS1 bench | baseline | now | Δ |
|---|---|---|---|
| open / `bag_info` | 2.79 ms | 0.05 ms | **−98 %** |
| read, 2 s window | 3.26 ms | 0.25 ms | **−92 %** |
| read, full | 5.41 ms | 2.97 ms | −45 % |
| read, two topics | 5.37 ms | 2.98 ms | −45 % |

Also: a pull iterator (`Ros1Reader::raw_iter`) shares the chunk-cache helper
with the callback path; header scanning on the hot path is allocation-free.

### 2.2 MESSAGE-compressed bags (accuracy)
rosbag2 `compression_mode: MESSAGE` stores each payload as a zstd frame.
`Reader` now decodes transparently for **both** sqlite and MCAP-directory
bags (our own `Writer` can produce either) through one `MessageDecoder` with a
reused zstd context pre-sized from the frame header; a payload that is not a
zstd frame is an error, never passed through. FILE-mode bags remain
unsupported on read (needs a temp-dir policy).

### 2.3 Parallel, order-preserving chunk compression (writers)
Chunks are independent, so finished chunks are compressed on the `rayon` pool
while the caller keeps appending, and written back strictly in order with
bounded in-flight memory.

* **MCAP**: own chunked encoder (`mcap/chunked.rs`) replaces the `mcap`
  crate's inline compressor (which also spins up a fresh zstd thread pool per
  chunk). zstd write 83 → 33 ms (**−61 %**, same-session criterion A/B).
  Output validated by the `mcap` crate, our reader, and Python `mcap` with CRC
  validation and index-driven reads on a real 1.27 GB dataset.
* **ROS1**: bz2/lz4 chunks compressed in parallel. Converting a real 1.27 GB
  bag to bz2: ~25 MiB/s → **163 MiB/s (6.5×)**, digest-identical to the source
  (`bag_verify`); lz4 runs at ~1.9 GB/s.

### 2.4 sqlite write pages
Large blobs spill into overflow-page chains at the default 4 KiB page size.
Controlled microbenchmark (same engine/pragmas, fixture-like payload mix, 8
alternating reps): median write 56.2 ms @4 KiB → 31.5 @16 → **27.4 @32 (−51 %)**
→ 24.2 @64 KiB; file +4 %. 32 KiB chosen (64 KiB would add a ~512 KB floor to
tiny bags). Reads unaffected within noise (open −4.6 %, full flat, two-topic
+5 %, window flat).

### 2.5 Catalog
`lifecycle_msgs` (4 types) and `actionlib_msgs` (3, ROS1) appended to
`catalog_ros2.msgs` (168 types, append-only — existing entries untouched). The
ROS1 md5sums computed for `GoalID`, `GoalStatus`, `GoalStatusArray` equal the
known ROS distribution values, which confirms both the definitions and the md5
implementation. The catalog-wide reference-encoder test round-trips every type
in both directions.

### 2.6 VRS accuracy
* `VrsReader::verify_record_chain()` walks the headers and checks
  `record_size`, stream and `previous_record_size` against the index; reported
  by `vrs_info --verify-index`. Intact on the real LaMaria file; a tampered
  header is located exactly (test).
* `AriaRecording::time_domain_map()` / `TimeDomainMap::monotonic_to_real_ns`:
  opt-in device-clock → UTC conversion from stream 285-1, with exact integer
  interpolation (offsets are ~1.7e18 ns, beyond `f64`). Real-file test: the
  converted start time lands on the recording's `capture_time_epoch` tag.
  Nothing applies it automatically.
* VRS-as-bag message `seq` is now the record's ordinal within its stream
  (`VrsReader::data_ordinal`). Previously a per-reader counter that never reset,
  so running the same query twice produced different bytes — found by the
  real-data pull/callback parity test.

### 2.7 VIO-facing API
`BagReader::iter(&Selection)` returns a `BagIterator` that pulls messages one
at a time in timestamp order from **any** container (ROS1, sqlite3, MCAP dir,
bare MCAP, VRS), with `next_into(&mut buf)` (buffer reuse), `next_message()`
and an `Iterator` impl. It yields exactly what `for_each` yields (tested on all
five containers and four filter shapes, and on a real MCAP and a real Aria
VRS). `BagReader::params()` returns a `ContainerParams`: ROS2 `metadata.yaml`
facts (storage, distro, compression, `custom_data`) or, for VRS, file tags, the
factory calibration as JSON, and typed camera/IMU configs.
Caveat for consumers: VRS orders by record-header time while emitted stamps are
capture times; sub-millisecond inversions are possible.

### 2.8 Smaller fixes (accuracy)
`CdrDeserializer` gained `read_i64`/`read_u64`; `read_sequence` caps its
pre-allocation at the remaining byte count (a corrupt 4 G length no longer
asks for gigabytes).

## 3. Measured scorecard (read side, quiet-enough machine)

| Bench | baseline | now | Δ |
|---|---|---|---|
| ros1 open | 2.79 ms | 0.05 ms | −98 % |
| ros1 read full / two topics | 5.4 ms | 3.0 ms | −45 % |
| ros1 read 2 s window | 3.26 ms | 0.25 ms | −92 % |
| sqlite open | 1.78 ms | 1.10 ms | −38 % |
| sqlite read full | 9.57 ms | 6.10 ms | −36 % |
| mcap read full | 5.54 ms | 4.68 ms | −15 % |
| vrs read full cursor | 4.16 ms | 2.65–3.28 ms | −21…−36 % |
| vrs read IMU-only | 3.27 ms | 2.32–2.36 ms | −28…−29 % |
| vrs copy trim | 2.17 ms | 1.39–1.64 ms | −24…−36 % |

Write-side figures are quoted from controlled comparisons (§2.3, §2.4), not
from the criterion `rosbag_write`/`vrs_write` rows: those write ~50 MB to the
external SSD and swing by up to 2× between identical runs on this machine
(load average 25–30 from other processes), so they cannot resolve effects this
small. An interleaved ABAB of uncompressed ROS1 writes before/after the
parallel writer showed no difference beyond that noise.

## 4. Deferred by design
* **ROS1 chunk-prefetch (parallel inflate) on read** — the lazy index removed
  the open/window cost; per-message decode of uncompressed chunks is not
  I/O-bound. Revisit if compressed ROS1 bags become a measured read bottleneck
  (no such file in the current datasets).
* **`.idl` schemas**, **FILE-mode sqlite compression on read** — unchanged
  (loud rejection / unsupported).
* **Applying the 285-1 clock map by default** — would silently change every
  consumer's timestamps; kept opt-in.
* **MCAP chunk-size tuning** — larger chunks speed up the old in-encoder
  threading (83 → 73 ms @16 MiB, → 51 ms @32 MiB) but raise reader prefetch
  memory 4–8×; superseded by §2.3.

## 5. Acceptance
`cargo clippy -p apex-io --all-features --all-targets -- -D warnings` clean in
all feature combinations; 749 tests pass with all features (629 rosbag-only);
`scripts/dataset_sweep.sh` (MCAP, 76 files) and the VRS checks
(`vrs_info --verify-index`, `scripts/vrs_crosscheck.py` against Meta's reader)
were run before this pass; real-data tests are env-gated (`APEX_VRS_SAMPLE`,
`APEX_BENCH_MCAP`).
