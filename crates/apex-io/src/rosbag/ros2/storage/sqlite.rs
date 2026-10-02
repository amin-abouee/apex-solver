//! SQLite3 storage backend implementation

use super::StorageReader;
use crate::rosbag::error::{BagError, ReaderError, Result};
use crate::rosbag::types::{
    Connection, Message, MessageDefinition, MessageDefinitionFormat, RawMessage,
};
use rusqlite::Connection as SqliteConnection;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// SQLite3 storage reader implementation
pub struct SqliteReader {
    /// Database file paths
    db_paths: Vec<PathBuf>,
    /// Database connections (one per file)
    connections: Vec<SqliteConnection>,
    /// Topic connections from metadata
    topic_connections: Vec<Connection>,
    /// Schema version detected from database
    schema_version: u32,
    /// Message type definitions
    message_definitions: HashMap<String, MessageDefinition>,
    /// Whether the reader is currently open
    is_open: bool,
}

impl SqliteReader {
    /// Create a new SQLite reader
    pub fn new(paths: Vec<&Path>, connections: Vec<Connection>) -> Result<Self> {
        let db_paths = paths.iter().map(|p| p.to_path_buf()).collect();
        Ok(Self {
            db_paths,
            connections: Vec::new(),
            topic_connections: connections,
            schema_version: 0,
            message_definitions: HashMap::new(),
            is_open: false,
        })
    }

    /// Detect the schema version from the database
    fn detect_schema_version(conn: &SqliteConnection) -> Result<u32> {
        let mut stmt = conn
            .prepare("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='schema'")?;
        let schema_table_exists: i32 = stmt.query_row([], |row| row.get(0))?;

        if schema_table_exists > 0 {
            let mut stmt = conn.prepare("SELECT schema_version FROM schema")?;
            let version: i32 = stmt.query_row([], |row| row.get(0))?;
            Ok(version as u32)
        } else {
            let mut stmt = conn.prepare("PRAGMA table_info(topics)")?;
            let rows = stmt.query_map([], |row| {
                let column_name: String = row.get(1)?;
                Ok(column_name)
            })?;

            let mut has_qos_profiles = false;
            for row in rows {
                if row? == "offered_qos_profiles" {
                    has_qos_profiles = true;
                    break;
                }
            }

            Ok(if has_qos_profiles { 2 } else { 1 })
        }
    }

    /// Load message definitions from the database (schema version 4+)
    fn load_message_definitions(
        &self,
        conn: &SqliteConnection,
    ) -> Result<HashMap<String, MessageDefinition>> {
        if self.schema_version < 4 {
            return Ok(HashMap::new());
        }

        let mut stmt = conn.prepare(
            "SELECT topic_type, encoding, encoded_message_definition, type_description_hash
             FROM message_definitions ORDER BY id",
        )?;

        let rows = stmt.query_map([], |row| {
            let topic_type: String = row.get(0)?;
            let encoding: String = row.get(1)?;
            let definition: String = row.get(2)?;
            let _hash: String = row.get(3)?;
            Ok((topic_type, encoding, definition))
        })?;

        let mut definitions = HashMap::new();
        for row in rows {
            let (topic_type, encoding, definition) = row?;

            let format = match encoding.as_str() {
                "ros2msg" => MessageDefinitionFormat::Msg,
                "ros2idl" => MessageDefinitionFormat::Idl,
                _ => MessageDefinitionFormat::None,
            };

            definitions.insert(
                topic_type,
                MessageDefinition {
                    format,
                    data: definition,
                },
            );
        }

        Ok(definitions)
    }
}

impl StorageReader for SqliteReader {
    fn open(&mut self) -> Result<()> {
        if self.is_open {
            return Ok(());
        }

        for path in &self.db_paths {
            let conn = SqliteConnection::open_with_flags(
                path,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
            )?;

            // Read-path tuning: mmap mapped pages for the keyset pagination
            // scans and a roomy page cache. Read-only, so durability is
            // untouched.
            for pragma in ["PRAGMA mmap_size = 268435456", "PRAGMA cache_size = -65536"] {
                if let Err(e) = conn.execute_batch(pragma) {
                    tracing::debug!("sqlite pragma '{pragma}' failed on {}: {e}", path.display());
                }
            }

            {
                let mut stmt = conn.prepare(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name IN ('messages', 'topics')"
                )?;
                let table_count: i32 = stmt.query_row([], |row| row.get(0))?;

                if table_count != 2 {
                    return Err(ReaderError::generic(format!(
                        "Database {} is missing required tables",
                        path.display()
                    )));
                }
            }

            self.connections.push(conn);
        }

        if !self.connections.is_empty() {
            let last_conn_idx = self.connections.len() - 1;
            let schema_version = Self::detect_schema_version(&self.connections[last_conn_idx])?;
            self.schema_version = schema_version;

            let definitions = self.load_message_definitions(&self.connections[last_conn_idx])?;
            self.message_definitions = definitions;
        }

        self.is_open = true;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        if !self.is_open {
            return Ok(());
        }

        self.connections.clear();
        self.message_definitions.clear();
        self.is_open = false;
        Ok(())
    }

    fn get_definitions(&self) -> Result<HashMap<String, MessageDefinition>> {
        if !self.is_open {
            return Err(ReaderError::BagNotOpen);
        }
        Ok(self.message_definitions.clone())
    }

    fn messages_filtered(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Box<dyn Iterator<Item = Result<Message>> + '_>> {
        let raw = self.raw_messages_filtered(connections, start, stop)?;
        Ok(Box::new(raw.map(|r| {
            r.map(|m| Message {
                topic: m.connection.topic.clone(),
                connection: m.connection,
                timestamp: m.timestamp,
                data: m.raw_data,
            })
        })))
    }

    fn is_open(&self) -> bool {
        self.is_open
    }

    fn raw_messages(&self) -> Result<Box<dyn Iterator<Item = Result<RawMessage>> + '_>> {
        self.raw_messages_filtered(None, None, None)
    }

    fn raw_messages_filtered(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Box<dyn Iterator<Item = Result<RawMessage>> + '_>> {
        if !self.is_open {
            return Err(ReaderError::BagNotOpen);
        }
        let topics = connections
            .filter(|c| !c.is_empty())
            .map(|c| c.iter().map(|c| c.topic.clone()).collect());
        Ok(Box::new(SqliteMessageIter::new(self, topics, start, stop)))
    }

    fn read_raw_messages_batch(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Vec<RawMessage>> {
        self.raw_messages_filtered(connections, start, stop)?
            .collect()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl SqliteReader {
    /// Get all topics and their message counts directly from the database
    pub fn get_topics_from_database(&self) -> Result<Vec<Connection>> {
        if self.connections.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_connections = Vec::new();

        for db_conn in &self.connections {
            // One GROUP BY over `messages` counts every topic in a single
            // pass; the old form ran one COUNT(*) query per topic, so a bag
            // with 94 topics paid 94 table scans on every open.
            let mut stmt = db_conn.prepare(
                "SELECT t.id, t.name, t.type, t.serialization_format, t.offered_qos_profiles, \
                 COUNT(m.id) \
                 FROM topics t LEFT JOIN messages m ON m.topic_id = t.id \
                 GROUP BY t.id ORDER BY t.id",
            )?;

            let topic_rows = stmt.query_map([], |row| {
                let id: i32 = row.get(0)?;
                let name: String = row.get(1)?;
                let message_type: String = row.get(2)?;
                let serialization_format: String = row.get(3)?;
                let offered_qos_profiles: String = row.get(4)?;
                let message_count: i64 = row.get(5)?;
                Ok((
                    id,
                    name,
                    message_type,
                    serialization_format,
                    offered_qos_profiles,
                    message_count,
                ))
            })?;

            for topic_result in topic_rows {
                let (_topic_id, name, message_type, serialization_format, _qos_profiles, count) =
                    topic_result?;

                let connection = Connection {
                    id: (all_connections.len() + 1) as u32,
                    topic: name,
                    message_type,
                    message_definition: MessageDefinition::default(),
                    type_description_hash: String::new(),
                    message_count: count.max(0) as u64,
                    serialization_format,
                    offered_qos_profiles: Vec::new(),
                };

                all_connections.push(connection);
            }
        }

        Ok(all_connections)
    }
}

/// Maximum rows fetched per page.
const PAGE_ROWS: usize = 2048;
/// Stop filling a page once it holds this many payload bytes, so a page of
/// camera frames stays small.
const PAGE_BYTES: usize = 16 * 1024 * 1024;

/// Streaming, bounded-memory message iterator over the db files of a bag.
///
/// Pages through each file's `messages` with a `(timestamp, id)` keyset
/// instead of materialising the result set, and merges the files by their
/// head keys, so messages come out in global timestamp order whatever the
/// number of files. Memory is one page per file, whatever the bag size.
struct SqliteMessageIter<'a> {
    reader: &'a SqliteReader,
    topics: Option<Vec<String>>,
    start: i64,
    stop: i64,
    /// Per-file state, built lazily when a file is first visited.
    states: Vec<Option<FileState>>,
    failed: bool,
}

struct FileState {
    topic_map: HashMap<i64, std::sync::Arc<Connection>>,
    /// `topic_id IN (...)` clause, empty when every topic is wanted.
    topic_clause: String,
    last: Option<(i64, i64)>,
    /// Page of fetched messages with their `(timestamp, id)` sort keys.
    page: std::collections::VecDeque<((i64, i64), Result<RawMessage>)>,
    done: bool,
}

impl<'a> SqliteMessageIter<'a> {
    fn new(
        reader: &'a SqliteReader,
        topics: Option<Vec<String>>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Self {
        Self {
            reader,
            topics,
            start: start.map_or(i64::MIN, |s| i64::try_from(s).unwrap_or(i64::MAX)),
            stop: stop.map_or(i64::MAX, |s| i64::try_from(s).unwrap_or(i64::MAX)),
            states: Vec::new(),
            failed: false,
        }
    }

    fn init_file(&self, db: &SqliteConnection) -> Result<FileState> {
        let mut topic_map = HashMap::new();
        let mut wanted_ids = Vec::new();
        let mut stmt = db.prepare("SELECT id, name, type, serialization_format FROM topics")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;
        for row in rows {
            let (id, name, message_type, serialization_format) = row?;
            if self.topics.as_ref().is_some_and(|t| !t.contains(&name)) {
                continue;
            }
            let connection = self
                .reader
                .topic_connections
                .iter()
                .find(|c| c.topic == name)
                .cloned()
                .unwrap_or_else(|| Connection {
                    id: u32::try_from(id).unwrap_or(0),
                    topic: name,
                    message_type,
                    message_definition: MessageDefinition::default(),
                    type_description_hash: String::new(),
                    message_count: 0,
                    serialization_format,
                    offered_qos_profiles: Vec::new(),
                });
            wanted_ids.push(id.to_string());
            topic_map.insert(id, std::sync::Arc::new(connection));
        }
        let topic_clause = if self.topics.is_some() {
            if wanted_ids.is_empty() {
                // None of the requested topics live in this file.
                "AND 0".to_string()
            } else {
                format!("AND topic_id IN ({})", wanted_ids.join(","))
            }
        } else {
            String::new()
        };
        Ok(FileState {
            topic_map,
            topic_clause,
            last: None,
            page: std::collections::VecDeque::new(),
            done: false,
        })
    }

    /// Fetch the next page of file `idx` into its queue.
    fn fill_page(&mut self, idx: usize) -> Result<()> {
        let db = &self.reader.connections[idx];
        let Some(state) = self.states[idx].as_mut() else {
            return Ok(());
        };
        let (last_ts, last_id) = state.last.unwrap_or((i64::MIN, i64::MIN));
        let sql = format!(
            "SELECT id, topic_id, timestamp, data FROM messages \
             WHERE (timestamp > ?1 OR (timestamp = ?1 AND id > ?2)) \
             AND timestamp >= ?3 AND timestamp < ?4 {} \
             ORDER BY timestamp, id LIMIT {PAGE_ROWS}",
            state.topic_clause
        );
        let mut stmt = db.prepare_cached(&sql)?;
        let mut rows = stmt.query((last_ts, last_id, self.start, self.stop))?;
        let mut fetched = 0usize;
        let mut bytes = 0usize;
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let topic_id: i64 = row.get(1)?;
            let timestamp: i64 = row.get(2)?;
            let data: Vec<u8> = row.get(3)?;
            state.last = Some((timestamp, id));
            fetched += 1;
            bytes += data.len();
            if let Some(connection) = state.topic_map.get(&topic_id) {
                state.page.push_back((
                    (timestamp, id),
                    Ok(RawMessage {
                        connection: connection.clone(),
                        timestamp: timestamp as u64,
                        raw_data: data,
                    }),
                ));
            }
            if bytes >= PAGE_BYTES {
                break;
            }
        }
        if fetched == 0 {
            state.done = true;
        }
        Ok(())
    }

    /// Make sure file `idx` has a state and a filled page. After this
    /// returns `Ok`, the file's head message is available unless the file is
    /// done.
    fn ensure_head(&mut self, idx: usize) -> Result<()> {
        if self.states.len() <= idx {
            self.states.resize_with(idx + 1, || None);
        }
        if self.states[idx].is_none() {
            self.states[idx] = Some(self.init_file(&self.reader.connections[idx])?);
        }
        let Some(state) = self.states[idx].as_mut() else {
            return Ok(());
        };
        if state.page.is_empty() && !state.done {
            self.fill_page(idx)?;
        }
        Ok(())
    }

    /// The `(timestamp, id)` sort key of file `idx`'s head message.
    fn head_key(&self, idx: usize) -> Option<(i64, i64)> {
        self.states
            .get(idx)?
            .as_ref()?
            .page
            .front()
            .map(|(key, _)| *key)
    }
}

impl Iterator for SqliteMessageIter<'_> {
    type Item = Result<RawMessage>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        // K-way merge by head key: pick the file whose front message has the
        // smallest (timestamp, id); earlier files win ties so equal
        // timestamps interleave in file order.
        let mut best: Option<(usize, i64, i64)> = None;
        for idx in 0..self.reader.connections.len() {
            if let Err(e) = self.ensure_head(idx) {
                self.failed = true;
                return Some(Err(e));
            }
            let Some((ts, id)) = self.head_key(idx) else {
                continue;
            };
            if best.is_none_or(|(_, bts, bid)| ts < bts || (ts == bts && id < bid)) {
                best = Some((idx, ts, id));
            }
        }
        let (idx, _, _) = best?;
        // `best` came from a checked head, so the pop cannot come back empty.
        self.states
            .get_mut(idx)
            .and_then(|s| s.as_mut())
            .and_then(|s| s.page.pop_front())
            .map(|(_, msg)| msg)
    }
}

/// SQLite storage writer implementation
pub struct SqliteWriter {
    /// Path to the database file
    db_path: PathBuf,
    /// Database connection
    connection: Option<SqliteConnection>,
    /// Whether compression is enabled (reserved for future use)
    _compression_mode: crate::rosbag::types::CompressionMode,
    /// Whether the writer is currently open
    is_open: bool,
    /// Connection ID mapping: topic -> database topic_id
    topic_id_map: HashMap<String, i32>,
    /// Messages inserted since the last COMMIT.
    pending: usize,
    /// Messages per transaction.
    commit_interval: usize,
}

impl SqliteWriter {
    fn begin(&mut self) -> Result<()> {
        if let Some(conn) = &self.connection {
            conn.execute_batch("BEGIN")?;
        }
        self.pending = 0;
        Ok(())
    }

    fn commit(&mut self) -> Result<()> {
        if let Some(conn) = &self.connection
            && !conn.is_autocommit()
        {
            conn.execute_batch("COMMIT")?;
        }
        self.pending = 0;
        Ok(())
    }

    /// Create a new SQLite writer
    pub fn new(
        path: &Path,
        compression_mode: crate::rosbag::types::CompressionMode,
    ) -> Result<Self> {
        if compression_mode == crate::rosbag::types::CompressionMode::Storage {
            return Err(BagError::writer(
                "SQLite3 writer does not support storage-side compression",
            ));
        }

        let file_name = path
            .file_name()
            .ok_or_else(|| BagError::writer("bag path has no file name component"))?
            .to_string_lossy();
        let db_path = path.join(format!("{file_name}.db3"));

        Ok(Self {
            db_path,
            connection: None,
            _compression_mode: compression_mode,
            is_open: false,
            topic_id_map: HashMap::new(),
            pending: 0,
            commit_interval: 20_000,
        })
    }

    /// Create the database schema
    fn create_schema(&self) -> Result<()> {
        let conn = self.connection.as_ref().ok_or(BagError::BagNotOpen)?;

        let schema = r#"
            CREATE TABLE schema(
                schema_version INTEGER PRIMARY KEY,
                ros_distro TEXT NOT NULL
            );
            CREATE TABLE metadata(
                id INTEGER PRIMARY KEY,
                metadata_version INTEGER NOT NULL,
                metadata TEXT NOT NULL
            );
            CREATE TABLE topics(
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                serialization_format TEXT NOT NULL,
                offered_qos_profiles TEXT NOT NULL,
                type_description_hash TEXT NOT NULL
            );
            CREATE TABLE message_definitions(
                id INTEGER PRIMARY KEY,
                topic_type TEXT NOT NULL,
                encoding TEXT NOT NULL,
                encoded_message_definition TEXT NOT NULL,
                type_description_hash TEXT NOT NULL
            );
            CREATE TABLE messages(
                id INTEGER PRIMARY KEY,
                topic_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                data BLOB NOT NULL
            );
            INSERT INTO schema(schema_version, ros_distro) VALUES (4, 'rosbags');
        "#;

        conn.execute_batch(schema)?;
        Ok(())
    }
}

impl crate::rosbag::storage::StorageWriter for SqliteWriter {
    fn open(&mut self) -> Result<()> {
        if self.is_open {
            return Err(BagError::BagAlreadyOpen);
        }

        let connection = SqliteConnection::open(&self.db_path)?;
        // A freshly written bag needs no crash safety: skip the journal and
        // fsyncs, keep a large page cache, and insert inside one long
        // transaction. The timestamp index is built once at close.
        connection.execute_batch(
            "PRAGMA page_size=32768; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; \
             PRAGMA locking_mode=EXCLUSIVE; PRAGMA cache_size=-65536;",
        )?;
        self.connection = Some(connection);

        self.create_schema()?;
        self.begin()?;

        self.is_open = true;
        Ok(())
    }

    fn close(&mut self, version: u32, metadata: &str) -> Result<()> {
        if !self.is_open {
            return Ok(());
        }

        if let Some(conn) = &self.connection {
            conn.execute(
                "INSERT INTO metadata(metadata_version, metadata) VALUES (?1, ?2)",
                (version, metadata),
            )?;
            conn.execute_batch("CREATE INDEX timestamp_idx ON messages (timestamp ASC);")?;
        }
        self.commit()?;

        self.connection = None;
        self.is_open = false;
        self.topic_id_map.clear();

        Ok(())
    }

    fn add_msgtype(&mut self, connection: &Connection) -> Result<()> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }

        let conn = self.connection.as_ref().ok_or(BagError::BagNotOpen)?;

        let encoding = match connection.message_definition.format {
            MessageDefinitionFormat::Msg => "ros2msg",
            MessageDefinitionFormat::Idl => "ros2idl",
            MessageDefinitionFormat::None => "ros2msg",
        };

        conn.execute(
            "INSERT INTO message_definitions(topic_type, encoding, encoded_message_definition, type_description_hash) VALUES (?1, ?2, ?3, ?4)",
            (
                &connection.message_type,
                encoding,
                &connection.message_definition.data,
                &connection.type_description_hash,
            ),
        )?;

        Ok(())
    }

    fn add_connection(
        &mut self,
        connection: &Connection,
        offered_qos_profiles: &str,
    ) -> Result<()> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }

        let conn = self.connection.as_ref().ok_or(BagError::BagNotOpen)?;

        conn.execute(
            "INSERT INTO topics(name, type, serialization_format, offered_qos_profiles, type_description_hash) VALUES (?1, ?2, ?3, ?4, ?5)",
            (
                &connection.topic,
                &connection.message_type,
                &connection.serialization_format,
                offered_qos_profiles,
                &connection.type_description_hash,
            ),
        )?;

        let topic_id = conn.last_insert_rowid() as i32;
        self.topic_id_map.insert(connection.topic.clone(), topic_id);

        Ok(())
    }

    fn write(&mut self, connection: &Connection, timestamp: u64, data: &[u8]) -> Result<()> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }

        let topic_id = self
            .topic_id_map
            .get(&connection.topic)
            .ok_or_else(|| BagError::connection_not_found(&connection.topic))?;

        let conn = self.connection.as_ref().ok_or(BagError::BagNotOpen)?;

        conn.prepare_cached("INSERT INTO messages(topic_id, timestamp, data) VALUES (?1, ?2, ?3)")?
            .execute((topic_id, timestamp as i64, data))?;

        self.pending += 1;
        if self.pending >= self.commit_interval {
            self.commit()?;
            self.begin()?;
        }
        Ok(())
    }

    fn set_commit_interval(&mut self, messages: usize) {
        self.commit_interval = messages.max(1);
    }

    fn is_open(&self) -> bool {
        self.is_open
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    use super::StorageReader;
    use super::*;
    use crate::rosbag::storage::StorageWriter;

    const SQLITE3_BAG: &str = "tests/test_bags/test_bag_sqlite3";

    fn open_test_reader() -> std::result::Result<SqliteReader, Box<dyn std::error::Error>> {
        use crate::rosbag::metadata::BagMetadata;
        let meta = BagMetadata::from_file(format!("{SQLITE3_BAG}/metadata.yaml"))?;
        let paths: Vec<std::path::PathBuf> = meta
            .info()
            .relative_file_paths
            .iter()
            .map(|p| std::path::PathBuf::from(SQLITE3_BAG).join(p))
            .collect();
        let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();
        let mut reader = SqliteReader::new(path_refs, vec![])?;
        reader.open()?;
        // populate topic_connections from the database so message queries work
        if let Ok(connections) = reader.get_topics_from_database() {
            reader.topic_connections = connections;
        }
        Ok(reader)
    }

    #[test]
    fn test_sqlite_reader_creation() -> TestResult {
        let reader = SqliteReader::new(vec![], vec![]);
        assert!(reader.is_ok());
        let reader = reader?;
        assert!(!reader.is_open());
        Ok(())
    }

    #[test]
    fn test_sqlite_reader_open_close() -> TestResult {
        let mut reader = SqliteReader::new(vec![], vec![])?;
        assert!(!reader.is_open());

        reader.open()?;
        assert!(reader.is_open());

        reader.close()?;
        assert!(!reader.is_open());
        Ok(())
    }

    #[test]
    fn sqlite_reader_get_definitions() -> TestResult {
        let reader = open_test_reader()?;
        let defs = reader.get_definitions()?;
        assert!(!defs.is_empty());
        Ok(())
    }

    #[test]
    fn sqlite_reader_as_any() -> TestResult {
        let reader = open_test_reader()?;
        let any = reader.as_any();
        assert!(any.is::<SqliteReader>());
        Ok(())
    }

    #[test]
    fn sqlite_reader_raw_messages_unfiltered() -> TestResult {
        let reader = open_test_reader()?;
        let count = reader.raw_messages()?.filter_map(|r| r.ok()).count();
        assert_eq!(count, 188);
        Ok(())
    }

    #[test]
    fn sqlite_reader_raw_messages_filtered_by_time() -> TestResult {
        let reader = open_test_reader()?;
        let all_msgs: Vec<_> = reader.raw_messages()?.filter_map(|r| r.ok()).collect();
        let min_ts = all_msgs.iter().map(|m| m.timestamp).min().ok_or("empty")?;
        let max_ts = all_msgs.iter().map(|m| m.timestamp).max().ok_or("empty")?;
        let mid = (min_ts + max_ts) / 2;

        let half_count = reader
            .raw_messages_filtered(None, Some(min_ts), Some(mid))?
            .filter_map(|r| r.ok())
            .count();
        assert!(half_count > 0 && half_count < 188);
        Ok(())
    }

    #[test]
    fn sqlite_reader_read_raw_messages_batch_full() -> TestResult {
        let reader = open_test_reader()?;
        let batch = reader.read_raw_messages_batch(None, None, None)?;
        assert_eq!(batch.len(), 188);
        Ok(())
    }

    #[test]
    fn sqlite_reader_messages_filtered_by_connection() -> TestResult {
        use crate::rosbag::metadata::BagMetadata;
        let meta = BagMetadata::from_file(format!("{SQLITE3_BAG}/metadata.yaml"))?;
        let paths: Vec<std::path::PathBuf> = meta
            .info()
            .relative_file_paths
            .iter()
            .map(|p| std::path::PathBuf::from(SQLITE3_BAG).join(p))
            .collect();
        let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();

        // Use apex_io Reader to get real connections
        use crate::rosbag::ros2::reader::Reader;
        let mut reader = Reader::new(SQLITE3_BAG)?;
        reader.open()?;
        let conns: Vec<_> = reader
            .connections()
            .iter()
            .filter(|c| c.topic == "/test/std_msgs/string")
            .cloned()
            .collect();

        let mut sqlite_reader = SqliteReader::new(path_refs, conns.clone())?;
        sqlite_reader.open()?;

        let count = sqlite_reader
            .messages_filtered(Some(&conns), None, None)?
            .filter_map(|r| r.ok())
            .count();
        assert_eq!(count, 2);
        Ok(())
    }

    #[test]
    fn sqlite_writer_new_creates_writer() -> TestResult {
        let dir = tempfile::tempdir()?;
        let writer = SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        assert!(!writer.is_open());
        Ok(())
    }

    #[test]
    fn sqlite_writer_open_close_cycle() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        writer.open()?;
        assert!(writer.is_open());
        writer.close(9, "{}")?;
        assert!(!writer.is_open());
        Ok(())
    }

    #[test]
    fn sqlite_writer_double_open_errors() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        writer.open()?;
        assert!(writer.open().is_err());
        Ok(())
    }

    #[test]
    fn sqlite_writer_add_msgtype_and_connection() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        writer.open()?;

        let conn = Connection {
            id: 1,
            topic: "/test".to_string(),
            message_type: "std_msgs/msg/String".to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        };

        writer.add_msgtype(&conn)?;
        writer.add_connection(&conn, "")?;
        Ok(())
    }

    #[test]
    fn sqlite_writer_write_message() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        writer.open()?;

        let conn = Connection {
            id: 1,
            topic: "/test".to_string(),
            message_type: "std_msgs/msg/String".to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        };

        writer.add_msgtype(&conn)?;
        writer.add_connection(&conn, "")?;
        writer.write(&conn, 1_000_000, &[0x00, 0x01, 0x00, 0x00])?;
        Ok(())
    }

    #[test]
    fn sqlite_writer_write_batch() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        writer.open()?;

        let conn = Connection {
            id: 1,
            topic: "/batch".to_string(),
            message_type: "std_msgs/msg/String".to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        };

        writer.add_msgtype(&conn)?;
        writer.add_connection(&conn, "")?;

        let msgs: Vec<(Connection, u64, Vec<u8>)> = vec![
            (conn.clone(), 100, vec![0x00, 0x01, 0x00, 0x00]),
            (conn.clone(), 200, vec![0x00, 0x01, 0x00, 0x00]),
        ];
        writer.write_batch(&msgs)?;
        Ok(())
    }

    #[test]
    fn sqlite_writer_write_batch_empty_is_noop() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        writer.open()?;
        let empty: Vec<(Connection, u64, Vec<u8>)> = vec![];
        writer.write_batch(&empty)?;
        Ok(())
    }

    #[test]
    fn sqlite_writer_write_without_open_errors() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        let conn = Connection {
            id: 1,
            topic: "/test".to_string(),
            message_type: "std_msgs/msg/String".to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        };
        assert!(writer.write(&conn, 0, &[]).is_err());
        Ok(())
    }

    #[test]
    fn sqlite_writer_add_msgtype_without_open_errors() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        let conn = Connection {
            id: 1,
            topic: "/test".to_string(),
            message_type: "std_msgs/msg/String".to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        };
        assert!(writer.add_msgtype(&conn).is_err());
        Ok(())
    }

    #[test]
    fn sqlite_writer_add_connection_without_open_errors() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        let conn = Connection {
            id: 1,
            topic: "/test".to_string(),
            message_type: "std_msgs/msg/String".to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        };
        assert!(writer.add_connection(&conn, "").is_err());
        Ok(())
    }

    #[test]
    fn sqlite_writer_as_any_returns_self() -> TestResult {
        let dir = tempfile::tempdir()?;
        let writer = SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        let any = <SqliteWriter as crate::rosbag::storage::StorageWriter>::as_any(&writer);
        assert!(any.is::<SqliteWriter>());
        Ok(())
    }

    #[test]
    fn sqlite_writer_close_when_not_open_is_noop() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        writer.close(9, "{}")?;
        assert!(!writer.is_open());
        Ok(())
    }

    #[test]
    fn sqlite_writer_write_unknown_connection_errors() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut writer =
            SqliteWriter::new(dir.path(), crate::rosbag::types::CompressionMode::None)?;
        writer.open()?;
        let unknown_conn = Connection {
            id: 999,
            topic: "/unknown".to_string(),
            message_type: "std_msgs/msg/String".to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        };
        assert!(writer.write(&unknown_conn, 0, &[]).is_err());
        Ok(())
    }
    #[test]
    fn multi_file_streams_in_global_timestamp_order() -> TestResult {
        use crate::rosbag::types::CompressionMode;
        let dir = tempfile::tempdir()?;

        // Two single-file bags with interleaved timestamps: file A holds
        // 100/102/104 ns, file B holds 101/103/105 ns on the same topic.
        let mut db_paths = Vec::new();
        let connection = Connection {
            id: 1,
            topic: "/t".into(),
            message_type: "std_msgs/msg/Int32".into(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".into(),
            offered_qos_profiles: Vec::new(),
        };
        for (name, times) in [("a", [100u64, 102, 104]), ("b", [101u64, 103, 105])] {
            let bag = dir.path().join(name);
            std::fs::create_dir(&bag)?;
            let mut w = SqliteWriter::new(&bag, CompressionMode::None)?;
            w.open()?;
            w.add_connection(&connection, "")?;
            for ts in times {
                w.write(&connection, ts, &ts.to_le_bytes())?;
            }
            w.close(4, "")?;
            db_paths.push(bag.join(format!("{name}.db3")));
        }

        let path_refs: Vec<&std::path::Path> = db_paths.iter().map(|p| p.as_path()).collect();
        let mut reader = SqliteReader::new(path_refs, vec![])?;
        reader.open()?;
        let timestamps: Vec<u64> = reader
            .raw_messages()?
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|m| m.timestamp)
            .collect();
        assert_eq!(timestamps, vec![100, 101, 102, 103, 104, 105]);
        Ok(())
    }
}
