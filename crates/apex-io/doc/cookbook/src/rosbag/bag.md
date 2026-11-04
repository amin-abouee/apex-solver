# One Reader and Writer for Every Container

`BagReader` and `BagWriter` (module `apex_io::rosbag`) are one reader and one writer over every
supported container, so a tool can copy, filter and convert without caring which backend sits
behind a path. The CLI tools (`bag_filter`, `bag_convert`, `bag_merge`, `bag_verify`) are
written against them.

| `BagKind` | On disk | Payload encoding |
|---|---|---|
| `Ros1Bag` | `*.bag` (ROS1 v2.0) | `ros1` |
| `Ros2Sqlite` | directory + `metadata.yaml` | `cdr` |
| `Ros2Mcap` | directory + `metadata.yaml` | `cdr` |
| `McapFile` | bare `*.mcap` (ROS2 or ROS1 profile) | per channel |

A VRS recording opens through the same reader (with the `vrs` feature): its streams appear as
ROS topics. See [VRS and LaMAria](../datasets/lamaria.md).

Both sides **stream**: messages reach your code as borrowed slices and the writers write
straight through, so memory use does not grow with the size of the bag.

## Reading

```rust
use apex_io::rosbag::{BagReader, Selection};

let mut bag = BagReader::open("recording")?;       // kind detected from the path
println!("{:?}: {} messages", bag.kind(), bag.message_count());

// Which messages: topics, and a half-open time window in nanoseconds.
let selection = Selection {
    include: vec!["/imu0".into()],
    start_ns: Some(bag.start_time() + 5_000_000_000),
    ..Selection::default()
};

// Push style: a callback per message, in timestamp order.
let connections = bag.selected_connections(&selection);
bag.for_each(&connections, &selection, |connection, timestamp_ns, payload| {
    // `payload` is the serialized message, borrowed for this call only.
    Ok(())
})?;

// Pull style: an iterator over every container, `next_into` to reuse one buffer.
let mut messages = bag.iter(&selection)?;
let mut buffer = Vec::new();
while let Some((connection_id, timestamp_ns)) = messages.next_into(&mut buffer)? {
    // `buffer` holds the payload
}
```

`Selection { include, exclude, start_ns, end_ns }`: an empty `include` means all topics;
`start_ns` is inclusive and `end_ns` exclusive. `BagReader::params()` returns the container's
sensor and container parameters (for a VRS file, the Aria sensor configuration).

`detect_kind(path)` identifies an existing bag from its file magic or directory layout;
`BagKind::from_output_path(path, default)` guesses the kind to *create* from an extension
(`.bag` is ROS1, `.mcap` a bare MCAP, anything else the default directory kind).

## Writing

```rust
use apex_io::rosbag::{BagKind, BagWriter, NewConnection, OutputCompression};

let mut out = BagWriter::create("merged", BagKind::Ros2Mcap, OutputCompression::Zstd)?;
// Declare each topic once (copy it from an input connection), keep the handle ...
let handle = out.add_connection(&NewConnection::from_connection(&connection))?;
// ... then write payloads verbatim.
out.write(handle, timestamp_ns, payload)?;
out.close()?;
```

`create` refuses an existing path. Payload bytes are written as they are, so the encoding must
match what the container holds (ROS1 for `.bag`, CDR for ROS2 directories). To change the
encoding, convert through [`transcode`](./overview.md#converting-between-ros1-and-ros2-transcode):
that is what `bag_convert` does. The MCAP writer compresses chunks in parallel and keeps their
order; the ROS1 writer compresses bz2 and lz4 chunks on all cores.

## Errors

Everything returns `BagResult<_>` = `Result<_, BagError>`, which the crate root wraps as
`IoError::Bag`. Bag-specific failures carry typed fields (`Unsupported { container, what }`,
`TooManyConnections`, `UnknownHandle`, …) and keep the cause of a container-level failure as
`source()`.
