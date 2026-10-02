# apex-io ROS1/ROS2/VRS: speed + accuracy review

Working notes for the optimization pass on `crates/apex-io` (branch
`feature/vrs-reader-writer`). Each section states the **issue** found in the
code (with file/line evidence at the time of review), the **solution**
applied, and the **measured result** where one exists. All numbers come from
the criterion benches introduced in this pass, on the synthetic VIO-like
fixture (2 × 10 Hz 320×240 cameras, 200 Hz IMU, 5 Hz 10k-point lidar, 20 s,
~48 MB per encoding) plus real-data benches over
`/Volumes/External/datasets` (uzh_fpv/tumvi MCAP, LaMaria VRS).

Baseline benches live in `crates/apex-io/benches/`; run with

```bash
cargo bench -p apex-io --features rosbag,vrs --bench rosbag_bench
APEX_BENCH_VRS=<file.vrs> cargo bench -p apex-io --features vrs --bench vrs_bench -- real_vrs
```

---

## 0. The goal

apex-io's bag stack exists so a consumer (a VIO front end, a dataset tool,
`bag_*`/`vrs_*` CLIs) can:

1. **open** any container — ROS1 `.bag`, ROS2 sqlite3/MCAP directory, bare
   `.mcap`, Meta VRS — behind one API (`BagReader`/`BagWriter`),
2. **read a chosen list of topics/streams time-ordered, as fast as possible**
   (cameras + IMU + lidar), with bounded memory whatever the file size,
3. **get params** — calibration/config — out of the container.

Everything below serves that: correctness first (silently wrong data is
worse than slow data), then per-message overhead, then overlap of I/O and
decompression with the consumer's work.

---

## 1. Benchmarks did not exist (commit `test(apex-io)`)

**Issue.** `crates/apex-io` had no `benches/`, no `[[bench]]` targets, no
criterion dependency. Every performance claim was unverifiable, and there
was no regression tripwire.

**Solution.**
- `criterion` (workspace 0.6) dev-dep + two bench targets gated by
  `required-features` (`rosbag_bench`, `vrs_bench`).
- Deterministic synthetic fixtures (`benches/helpers/`): ROS1-encoded
  payloads built with the crate's own `Ros1Serializer` + typed encoders,
  CDR twins via the schema transcoder, incompressible camera/lidar bytes
  and compressible IMU bytes via a fixed xorshift — so benches measure real
  payloads through the public API only.
- Real-data benches behind `APEX_BENCH_MCAP` / `APEX_BENCH_VRS` (skip when
  unset).
- Fixture scratch space lives under `<target>/apex-io-bench` (wiped per
  run, drop-guarded per iteration) — the system temp volume filled up once
  and the disk was nearly full, so benches must never write multi-GB
  fixtures to `/tmp` or leak them.

Groups: `rosbag_info` / `rosbag_read{full,two_topics,window}` /
`rosbag_write` / `rosbag_convert` / `rosbag_transcode` / `real_mcap`;
`vrs_open` / `vrs_read{full,imu_only,images_only,full_read_ahead}` /
`vrs_write` / `vrs_copy{full,trim}` / `real_vrs{open,read_full_raw,
read_full_raw_read_ahead,decode_window,decode_imu_window,imu_between_1s}`.

---

## 2. VRS: a fresh zstd context per record (commit `perf(apex-io)` — codec)

**Issue.** `vrs/codec.rs` decompressed every record with a brand-new
`zstd::stream::read::Decoder` (a fresh `DCtx`, hundreds of KB of setup)
and filled the output through `read_to_end`'s incremental growth; writing
used `zstd::bulk::compress` (fresh `CCtx`) per record. For the ~1 kHz IMU
streams with ~200-byte payloads this fixed per-record cost dominated
everything: 648k records in a 3.2 GB LaMaria file pay it 648k times.

**Solution.**
- `codec::Decompressor` / `codec::Compressor` own reusable
  `zstd::bulk::Decompressor` / `Compressor` contexts; `VrsReader`,
  `VrsWriter` and the copy path hold one each.
- Decompression is one `decompress_to_buffer` call into the pre-sized
  output buffer (the header announces the uncompressed size) instead of
  `read_to_end`.
- The one-shot free functions stay for rare records: the **index record**
  carries trailing bytes after the compressed frame, which the bulk API
  rejects; the hot path is intentionally strict (trailing data in a data
  record is corruption), the rare path tolerant.
- `RecordLayout::split_into` (caller-reused block vector) added alongside
  `split`.

**Measured (synthetic).** full cursor 4.13 → 2.63 ms (**−36 %**), IMU-only
3.32 → 2.32 ms (**−30 %**), copy 12.67 → 9.98 ms (−21 %). Stored payloads
(camera frames) unchanged, as expected — they never decompress.

---

## 3. VRS: copy recompressed everything; reads had no pipeline
(commits `perf(apex-io)` — passthrough, read-ahead)

**Issue (copy).** `vrs/copy.rs` decompressed every record and recompressed
it into the output even when source and target compression were identical
— the common case. Every byte of a 12 GB trim was paid twice.

**Issue (reads).** `VrsReader::next_record_into` does seek + read +
decompress **on the consumer's thread**, serially ahead of whatever the
consumer does with the record. A VIO loop that wants three image topics +
IMU pays that latency per record; nothing overlaps.

**Solution (copy).** `VrsReader::read_stored_record_into` hands out the
record as stored (`StoredRecord`); `VrsWriter::write_record_raw` writes an
already-stored body verbatim (keeping the `previous_record_size` chain and
index). The copy path picks the cheapest route per record:

- source compression == target → **verbatim passthrough** (no decompress,
  no recompress);
- source stored → compress directly (no decompress);
- otherwise → one decompress + recompress.

**Solution (read-ahead).** `vrs/readahead.rs`: `VrsReader::sequential(&filter)`
returns a `SequentialReader` that yields exactly the records of the
synchronous cursor loop (equivalence test included) but reads and
decompresses **ahead** on a worker thread with its own file handle:

- gated by a **record count and byte budget** (default 256 records /
  32 MiB in flight) so tiny IMU records batch by the hundreds while
  camera frames batch by a handful — memory stays bounded either way;
- **low-watermark refills**: dispatching on every pop degraded to
  one-record batches where the ~µs thread round trip dwarfed the work per
  record (measured 4× slower on a real file before the watermark);
- job channel bounded (backpressure), results channel unbounded (the
  worker never blocks on its way back to reading), so no send/receive
  cycle can deadlock;
- two real deadlock bugs were found and fixed during development:
  (a) bounded results channel + per-pop dispatch → mutual blocking
  (diagnosed by sampling the hung process);
  (b) `done` flips to true *inside* `dispatch` when the scan exhausts
  with an empty batch — the `done && inflight == 0` check ran *before*
  dispatch, so the consumer then blocked forever on a worker that would
  never send. The check is repeated after dispatch.

**Measured.** full copy 12.67 → 7.36 ms (**−42 %**), trim −26 %. Read-ahead
on the real 3.2 GB LaMaria file: 2383 ms → 614 ms after the watermark fix,
vs 577 ms synchronous with a no-op consumer on a page-cached file — the
~6 % overhead is the price of the pipeline when the consumer does nothing;
it turns into a win as soon as per-record consumer work or cold I/O
appears (the VIO case). Opt-in by design.

---

## 4. MCAP: every message copied twice; filters hashed per message
(commit `perf(apex-io)` — mcap drain)

**Issue.** In the indexed chunk drain (`mcap_format.rs`,
`next_inner_message`) each record was copied out of the chunk buffer
(`content.to_vec()`) and then the payload copied again
(`data = content[22..].to_vec()`) — two full payload copies per message,
i.e. ~96 MB of memcpy for a 48 MB scan. Topic filtering then cloned the
topic `Rc<str>` and hashed it into a `HashSet<String>` **per message**.
`McapStorageReader` re-aggregated the summary section (topic census +
stats) on every `messages_filtered`/`raw_messages_filtered` call. The
unindexed fallback walk read the whole record then copied the payload
again.

**Solution.**
- Message records parse **in place** (fields are plain little-endian
  integers at fixed offsets) and the payload copies **once**; a message
  shorter than its 22-byte fixed header is now an error instead of a
  panic (`content[22..]` used to panic on corrupt input).
- `channel_wanted` memoizes the topic-filter decision per channel ID —
  per-message cost drops to one small `BTreeMap<u16, bool>` lookup, and
  the old "unknown channel" error is preserved.
- Unindexed walk: reads the 22-byte header first and **skips the payload
  read entirely** for filtered-out messages.
- Summary aggregation cached in `open()` (`aggregated: Option<…>`), so
  the per-query cost is a clone-free borrow.

**Measured.** full scan 5.26 → 4.79 ms (−9 %), two-topic scan 5.24 →
4.53 ms (−13.5 %). (Same-session ros1/sqlite3 rows drifted +5–12 % on
untouched code — machine noise; final docs will re-measure on a quiet
machine.)

---

## 5. sqlite: 94 COUNT(*) queries per open, no read tuning, multi-file
order violated the contract (commit `perf(apex-io)` — sqlite)

**Issue.**
- `get_topics_from_database` ran one `SELECT COUNT(*) … WHERE topic_id = ?`
  **per topic**: a bag with 94 topics paid 94 scans of `messages` on every
  open.
- Read connections ran with default pragmas — no mmap, tiny page cache —
  so the keyset pagination of a full scan re-fetched pages through the
  buffer pool.
- `SqliteMessageIter` drained db3 file 0 completely, then file 1 …  A
  multi-file bag therefore did **not** iterate in global timestamp order,
  contradicting the documented `BagReader::for_each` contract (MCAP does
  merge; sqlite concatenated).
- Bag-level zstd (`CompressionMode::File`) read the whole storage file
  into RAM to compress it — a 12 GB bag needs 12 GB of heap.

**Solution.**
- One `GROUP BY` census (`topics LEFT JOIN messages … GROUP BY t.id`) —
  a single pass counts every topic.
- Read-only pragmas at open: `mmap_size = 256 MiB`, `cache_size = 64 MiB`
  (failures logged, never fatal).
- The iterator now keeps one keyset page **per file** and k-way merges by
  head key `(timestamp, id)` — earlier files win ties, so equal
  timestamps interleave in file order. Bounded memory: one page per file.
  Covered by a new test writing two interleaved single-file bags and
  asserting the merged order `100,101,102,103,104,105`.
- Bag-level zstd streams through `zstd::stream::Encoder` + `io::copy`
  with bounded buffers.

**Measured.** open 1.74 → 1.12 ms (**−35 %**), full scan **−44 %**
(9.99 ms at baseline → ~3.2 ms), window −41 %.

---

## 6. ROS1: unvalidated index metadata (commit in progress)

**Issue.** The ROS1 reader parsed `IndexData` without checking the `ver`
header field (the spec pins it to 0), and never cross-checked the bag
header's `conn_count`/`chunk_count` against what the index actually
holds — a truncated or foreign-tailed bag opened with silently wrong
counts. Open used the default 8 KiB `BufReader` while walking the whole
index tail in small reads.

**Decision on chunk prefetch.** The plan sketched a ROS1 chunk-prefetch
pool mirroring MCAP's. The benches say no: an uncompressed-chunk ROS1 scan
(5.45 ms) is already at parity with MCAP — the path is per-message-bound,
not I/O-bound, so a worker pool would add threads without a measurable
win. Skipping it keeps the reader single-threaded and predictable;
revisit only if compressed-chunk bags become a measured bottleneck.

**Solution (applied).** `ver` validation on `IndexData`, `conn_count` and
`chunk_count` cross-checks at open, 1 MiB `BufReader`.

## 7. CDR: missing 64-bit readers; unbounded sequence allocation
(commit in progress)

**Issue.** `CdrDeserializer` had no `read_i64`/`read_u64` (so typed
decoders for messages with 64-bit fields could not exist), and
`read_sequence` called `Vec::with_capacity(length)` on an attacker- or
corruption-controlled length — up to 4 G elements before the first
element read fails.

**Solution.** `FromBytes` for `i64`/`u64` + `read_i64`/`read_u64`;
`read_sequence` caps capacity at the remaining byte count (elements
cannot be smaller than one byte) — a corrupt length now fails on the
reads instead of over-allocating.

## 8. sqlite: MESSAGE-compressed bags read back garbage (commit in
progress)

**Issue.** `Writer` supports rosbag2-style MESSAGE compression: with
`CompressionMode::Message` + zstd it compresses every blob before INSERT.
The reader never decompressed — such bags iterate fine but every payload
is zstd garbage. Worse, nothing tells the reader this happened except
metadata.yaml (`compression_mode: MESSAGE`).

**Solution.** `SqliteReader::set_message_compression(CompressionFormat)`
fed from metadata (`compression_mode == "MESSAGE"`, format zstd) by
`Reader::open` via the existing downcast; the iterator decompresses blobs
with a reused bulk `Decompressor` as it pages them in. (FILE-mode bags —
`.db3.zstd` files — remain unsupported on read, as before.)

---

---

## 9. Remaining work, in detail

### 9.1 sqlite: finish the MESSAGE-compression read path — *in flight*

**What.** rosbag2 has two compression modes recorded in `metadata.yaml`:
`compression_mode: FILE` (the whole `.db3` is renamed `.db3.zstd`) and
`compression_mode: MESSAGE` (every blob in `messages.data` is a zstd
frame). apex-io's `Writer` has been able to produce MESSAGE-mode bags for
a while (`ros2/writer.rs` compresses each payload with a reused bulk
compressor before `storage.write`), but the reader never decoded them.

**Issue.** A MESSAGE-compressed bag iterates "successfully" — right
topics, right timestamps, right message counts — and hands out zstd
garbage as every payload. Nothing errors; the corruption is discovered
downstream (a CDR deserializer choking, or worse, silent nonsense). The
only place the fact is recorded is `metadata.yaml`
(`compression_mode: MESSAGE`, `compression_format: zstd`); the storage
layer itself cannot tell.

**Why it matters.** Bag-to-bag comparisons and round-trips
(`write_dummy_bag --compression`, cross-tool interop with rosbag2) silently
produce garbage payloads today. This is an accuracy hole, not a perf one.

**Solution (designed; ~80 % implemented).**
1. `SqliteReader` gains `message_compression: Option<CompressionFormat>`
   plus a setter `set_message_compression(...)` — done.
2. `SqliteMessageIter` decompresses each blob while paging it in
   (`decompress_message_blob`, streaming `zstd::stream::decode_all` — the
   frame is self-terminating and carries no length prefix, so the decoder
   sizes the output; a blob that does not start with a zstd frame is a
   corrupt bag and must error, never silently pass through) — done.
3. `Reader::open` feeds it from metadata via the existing
   `as_any()` downcast: `compression_mode == "message"` (case-insensitive,
   matching our writer's `as_str`), format `zstd`, anything else →
   `UnsupportedCompressionFormat` — **one compile error left**: the
   downcast needs `&mut` (`downcast_mut`) but `storage` is a shared
   `Box<dyn StorageReader>`; fix is to do the metadata→setter step on the
   `&mut storage` binding *before* it is moved into `self.storage`
   (a two-line reordering), or to have `create_storage_reader` take the
   compression hint as a parameter.
4. Round-trip test: write via `Writer` + `set_compression(Message, Zstd)`,
   read via `Reader`, assert payloads byte-identical; plus a negative test
   (blob with the zstd magic stripped must error, not yield garbage).

**Scope notes.** FILE-mode bags stay unsupported on read (as before) — the
storage file is literally renamed `.db3.zstd`, so `create_storage_reader`
never sees a valid sqlite file; supporting it means decompressing the file
to a temp location, which needs a temp-dir policy decision. MCAP needs
none of this: chunk compression is inside the container and already
decoded by the MCAP reader.

**Verification.** New unit test(s) in `sqlite.rs`; the existing
`write_dummy_bag` + `cli_roundtrip` suites must stay green.

---

### 9.2 Standard-message catalog vs ternaris/rosbags

**What.** `transcode/catalog_ros2.msgs` embeds 161 message definitions
across 18 packages, generated from a recent ROS 2 distribution, compiled
once into a `Registry`, and used by the schema-driven ROS1↔CDR transcoder,
by `BagWriter::add_connection` (fills definitions for standard types when
the source bag has none), by `bag_convert`/`bag_verify`, and by the VRS
Aria mapping. The `plan.rs` test proves every catalog type transcodes both
directions against independently generated reference encoders, and
`catalog.rs` self-checks closure under dependencies.

**Issue.** "All standard messages" is defined by what ships with ROS
distributions — and ternaris/rosbags (the reference implementation the
user pointed at) generates its type stores from exactly those
definitions. Diffing our catalog against rosbags' stores and the ROS core
message set shows three classes of gap:

1. **`lifecycle_msgs`** (ROS 2): `State`, `Transition`,
   `TransitionDescription`, `TransitionEvent` — emitted by every managed
   ROS 2 node; a recorded system with lifecycle nodes cannot be converted
   today (topic planned out with a warning).
2. **`actionlib_msgs`** (ROS 1): `GoalID`, `GoalStatus`,
   `GoalStatusArray` — every ROS 1 action server publishes these. ROS 1
   bags with actions fail conversion. (ROS 2 renamed the set to
   `action_msgs`, which we already have — the ROS 1 dialect names differ,
   so the catalog needs the ROS 1-only package too.)
3. **`std_srvs` service-event message types** (ROS 2): srv types appear
   in bags as `..._Event` messages (`std_srvs/srv/SetBool_Event` etc.);
   rosbags' stores include them.
4. Any **per-type drift** within covered packages (e.g. types added in
   newer distros like `nav_msgs/Goals`, which we already carry — the diff
   must be exhaustive, not assumed).

**Why it matters.** The user's stated goal is "support all standard
messages for both ros1 and ros2, checked with rosbags". Today a bag
containing any of the above converts with those topics skipped (warning +
exit code 2 with `--strict`), which is honest but incomplete.

**Planned solution.**
1. Generate the missing `.msg` texts from the ROS 2 / ROS 1 sources
   (mirroring how the existing catalog was produced), keeping the file's
   conventions (concatenated `MSG:` sections, dialect-neutral folding of
   `time`/`duration`/`Header`).
2. Append to `catalog_ros2.msgs` in dependency order; the closure
   self-check (`catalog.rs`) will immediately flag anything missing that
   the new types depend on.
3. Extend the ROS 1 md5 crosscheck table (`msgdef.rs` has known-md5
   assertions) with the new types — the md5 must match what `genmsg`
   produces, since ROS 1 connection headers carry it.
4. Both-dialect round-trip tests come free from the existing
   `matches_reference_encoders_for_every_catalog_type` test — it iterates
   every catalog type, so new types are automatically covered.

**Scope notes.** `.idl` (`ros2idl`/`omgidl`) schema parsing stays **out**:
ternaris supports it, but our transcoder is definition-text-based, IDL
parsing is a project of its own (comments, arrays with bounds, defaults,
annotations), and bags in the wild here (EuRoC, HILTI, TUM-VI, advio,
uzh_fpv) all carry `ros2msg` schemas. Rejection remains explicit
(`msgdef.rs` errors with "IDL message definitions are not supported") so
the failure is loud, and `get_definitions` preserves the raw IDL text for
consumers that can use it.

**Verification.** `cargo test -p apex-io --features rosbag` (catalog
closure + reference-encoder round trips run for free), plus a
diff-vs-ternaris check scripted once and recorded in the commit message.

---

### 9.3 VRS accuracy vs Meta's spec (`facebookresearch/vrs`)

Four gaps against the C++ reference, in priority order:

**9.3.1 Tag content types.**
*Issue.* VRS tags are typed (string, int, double, blob); our
`description.rs` decodes every value as a length-prefixed UTF-8 string
with `from_utf8_lossy`. A numeric or binary tag value therefore arrives
mangled (never rejected) — e.g. `device_serial`-style int tags become
lossy-decoded mojibake. Real Aria files we validated use string tags, so
tests pass; the spec does not promise that.
*Solution.* Parse the tag content type byte(s) per Meta's tag encoding,
represent values as an enum `TagValue { Text(String), Int(i64),
Double(f64), Blob(Vec<u8>) }` with `as_str()`/`as_i64()` accessors,
keeping `TagMap`'s string API working over `Text`/`Int`/`Double`
(formatted) so nothing breaks. Writer side gains typed setters (writes
strings as before).
*Verification.* Synthetic VRS files with typed tags round-trip; real
LaMaria files show unchanged values.

**9.3.2 `RecordType::Tags` records.**
*Issue.* Type-4 records are indexed but never parsed anywhere — their
payloads are invisible to `vrs_dump`/`vrs_info`.
*Solution.* Parse the tags-record payload into the same `TagMap` type and
surface via `VrsReader::record_tags(index)`; `vrs_dump` prints them.
*Verification.* Synthetic writer test (writer does not emit them yet —
test the parser against a hand-built payload) + real-file smoke check.

**9.3.3 `previous_record_size` chain validation.**
*Issue.* The 32-byte record header carries the previous record's size as
an integrity chain; we write it but never validate it on read. A torn
write or a doctored index could shift a record boundary and the reader
would only notice via size/stream mismatch — or not at all.
*Solution.* During `IndexSource::Rebuilt` (already a scan) and as a
`vrs_info --verify-index` option on the indexed path: walk sequentially,
assert each header's `previous_record_size` matches the last record's
size, report the first mismatch with both offsets. Strict per-record
enforcement on the hot read path stays off (it costs a seek-adjacent
extra read for random access; the chain is only checkable sequentially).
*Verification.* `index_matches_scan` extended; corrupted-copy unit test.

**9.3.4 DataLayout coverage + the 285-1 clock map.**
*Issue.* `datalayout.rs` decodes `DataPieceValue/Array/Vector<T>/string/
Vector<string>` + `Point3Df`. Meta's layouts also use multi-dim arrays
and nested constructs; unknown pieces are kept as `Unknown` (bytes
unreachable via the typed API) and undecodable layouts downgrade to a
warning at open. Separately, stream 285-1 (monotonic↔real-time map) is
decoded into `TimeDomainSample`s but applied nowhere — Meta's
`TimeQueryOptions`/`TimeDomain` semantics have no equivalent, and header
timestamps are treated as device time everywhere (true for Aria, not
guaranteed generally).
*Solution.* Extend the piece set driven by **real files first** (scan the
LaMaria corpus for `Unknown` pieces — the sweep script reports them;
implement what actually occurs, warn-and-skip what doesn't). Expose
`AriaRecording::time_domain_map() -> &[TimeDomainSample]` and a
`capture_to_realtime(ns) -> Option<i64>` helper; no default behavior
change, so VIO consumers opt in.
*Verification.* vrs_real_file suite + `vrs_crosscheck.py` against
`projectaria_tools` (values, timestamps, decoded pixels) on LaMaria.

---

### 9.4 Unified VIO-facing read API + params

**What.** The stated end goal: "open this mcap or vrs, read a list of
topics, get the params" — one API a VIO front end calls.

**Issue.** Today the unified layer exists (`BagReader::for_each` callback
over all five containers) but has three gaps for the VIO use case:
1. callback-style only — no way to own the loop's iteration state across
   re-entrant calls (VIO pipelines keep state between samples and want a
   `while let Some(msg) = it.next()` shape);
2. no params/calibration surface — VRS `calib_json` is an opaque string
   tag; camera/IMU config records need the typed Aria layer; ROS 2
   `metadata.yaml` `custom_data` has no accessor at the unified level;
3. no documented multi-topic time-ordered pull pattern with prefetch
   (the pieces exist: `Selection`, per-backend merge, `SequentialReader`
   for VRS — but they are separate knobs).

**Planned solution.**
1. A buffer-owning pull iterator on `BagReader`:
   `BagReader::iter(&Selection) -> Result<BagIterator>` with
   `next_into(&mut buf) -> Result<Option<MessageRef>>`, reusing each
   backend's merge (including the fixed sqlite merge and, for VRS, the
   cursor). The existing `for_each` stays (it is the zero-cost path for
   copy/convert tools); the iterator is the ergonomic VIO entry point.
2. `BagReader::params() -> ContainerParams`: an enum/struct with
   - `Vrs`: parsed `CameraCalibrations` JSON (intrinsics per camera via
     `apex-camera-models` types where they fit), per-sensor `CameraConfig`
     / `ImuConfig`, device serial/session;
   - `Ros2`: `metadata.yaml` `custom_data` map, per-connection QoS,
     storage plugin, compression;
   - `Ros1`: connection header definitions + md5s.
   Calibration JSON parsing stays best-effort: unknown fields pass
   through as raw JSON so nothing is lost.
3. A cookbook chapter ("Reading for VIO") showing the same code over an
   MCAP and a VRS file: open → select topics → time-ordered pull →
   params, plus the header-time vs capture-time ordering caveat
   (`vrs_bag.rs` orders by record-header time while emitting capture-ns
   stamps; sub-ms inversions are possible and must be tolerated or the
   consumer should sort by the emitted stamp).
4. Env-gated integration test running that exact flow over a real
   uzh_fpv MCAP and a real LaMaria VRS file.

**Scope notes.** No async runtime in scope: the thread-based
`SequentialReader` (9.x above, VRS today) is the overlap mechanism; MCAP
already prefetches chunk decompression; ROS1 stays synchronous by
measured decision (§6). A tokio façade can wrap the sync API later without
design changes.

**Verification.** New unit tests per accessor + the env-gated real-data
integration test + doc examples compiling as doctests.

---

### 9.5 Docs + final bench table

README/CHANGELOG per repo convention, with a re-measured table: every
bench re-run on a quiet machine (the commit-4/5 session showed ±5–12 %
drift on untouched paths), before/after columns from this document's
scorecard, the VRS real-file numbers, and the
`scripts/dataset_sweep.sh` + `vrs_sweep.sh` runs over
`/Volumes/External/datasets` as the end-to-end acceptance
(info/convert/verify/round-trip over every MCAP and VRS file, plus
`vrs_crosscheck.py` against Meta's reader).

---

### 9.6 Deferred by design (recorded decisions)

- **ROS1 chunk-prefetch pool** — benches show ROS1 scans are
  per-message-bound, not I/O-bound (§6); threads would add no measurable
  win. Revisit if compressed-chunk ROS1 bags become a real workload.
- **`.idl` schema parsing** — large surface, no bags in the current
  datasets carry them; explicit loud rejection today (§9.2).
- **FILE-mode sqlite compression on read** — needs a temp-dir policy
  (§9.1); MESSAGE-mode is the case our own writer produces.
- **Applying the 285-1 time-domain map to emitted stamps by default** —
  would silently change every consumer's timestamps; exposed as an
  opt-in query instead (§9.3.4).

## 10. Mid-flight state (uncommitted at the time of writing)

Commit 5 is landed; the sqlite MESSAGE-compression read (§9.1) is
~80 % implemented in the working tree: `SqliteReader` field + setter, blob
decompression in the iterator, metadata plumbing in `Reader::open` — with
one known compile error at `rosbag/ros2/reader.rs` (`downcast_mut` through
a shared `Box<dyn StorageReader>`; fix per §9.1 step 3) and the round-trip
tests still to add. `doc/perf_accuracy_review.md` (this file) is new and
untracked; it lands with the docs commit (§9.5), updated to final numbers.

## Scorecard (synthetic fixture, so far)

| Bench | Before | After | Δ |
|---|---|---|---|
| vrs read full cursor | 4.13 ms | 2.63 ms | −36 % |
| vrs read imu-only | 3.32 ms | 2.32 ms | −30 % |
| vrs copy full | 12.67 ms | 7.36 ms | −42 % |
| vrs read-ahead (real 3.2 GB) | 2383 ms | 614 ms | −74 % |
| mcap read full | 5.26 ms | 4.79 ms | −9 % |
| mcap read two topics | 5.24 ms | 4.53 ms | −13.5 % |
| sqlite open | 1.74 ms | 1.12 ms | −35 % |
| sqlite read full | 9.99 ms | ~3.2 ms | −68 % |
| ros1 read full | 5.45 ms | — | (deferred: not I/O-bound) |
