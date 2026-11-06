//! HTTP download with bounded waits, retries and a staged write.
//!
//! A dataset download runs inside tests and CI jobs that have no one watching.
//! Three properties keep it from wedging them:
//!
//! * every wait is bounded: a server that accepts the connection and then stops
//!   sending is abandoned after [`Policy::read`], instead of blocking the caller
//!   for as long as the process lives;
//! * transient failures (a dropped connection, a timeout, a 5xx) are retried a
//!   few times here, once, rather than in whichever test happened to notice;
//! * the body is streamed into `<name>.part` and renamed onto the final path
//!   only when complete, so the final path never holds a partial file and the
//!   whole body is never held in memory.

use std::ffi::OsString;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use tracing::warn;

use super::{DatasetError, DatasetResult};
use crate::error::ErrorLogging;

/// Longest wait for a TCP connection before an attempt is abandoned.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Longest a single read may block. This bounds *inactivity*, not the transfer:
/// a large file that keeps arriving never trips it.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Attempts per download (the first try included).
const ATTEMPTS: u32 = 3;
/// Pause before the second attempt; each further attempt waits proportionally longer.
const BACKOFF: Duration = Duration::from_secs(2);
/// Size of the buffer the body is streamed through.
const CHUNK: usize = 64 * 1024;

/// Limits applied to one download.
#[derive(Debug, Clone, Copy)]
pub(super) struct Policy {
    /// Longest wait for the connection to open.
    pub connect: Duration,
    /// Longest a single read may block.
    pub read: Duration,
    /// Attempts, the first try included.
    pub attempts: u32,
    /// Pause before the second attempt, multiplied for each further one.
    pub backoff: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            connect: CONNECT_TIMEOUT,
            read: READ_TIMEOUT,
            attempts: ATTEMPTS,
            backoff: BACKOFF,
        }
    }
}

/// Download `url` to `dest` under `policy`.
///
/// Transient failures are retried up to `policy.attempts` times in total; the
/// last error is returned if none succeeds. `dest` is replaced only by a
/// complete body: on any failure it is left exactly as it was.
pub(super) fn download(url: &str, dest: &Path, policy: &Policy) -> DatasetResult<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| DatasetError::write(parent, e).log())?;
    }

    // `timeout_per_read` is an idle limit: it restarts with every read, so a slow
    // but moving transfer of a large file is fine and a stalled one is not.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(policy.connect))
        .timeout_per_read(Some(policy.read))
        .timeout_per_write(Some(policy.read))
        .build()
        .into();
    let staged = staging_path(dest);

    let mut attempt = 1;
    loop {
        match fetch(&agent, url, &staged) {
            Ok(()) => return publish(&staged, dest),
            Err(failure) => {
                remove_staged(&staged);
                if !failure.transient || attempt >= policy.attempts {
                    return Err(failure.error.log());
                }
                warn!(
                    url,
                    attempt,
                    attempts = policy.attempts,
                    error = %failure.error,
                    "download failed, retrying"
                );
                std::thread::sleep(policy.backoff.saturating_mul(attempt));
                attempt += 1;
            }
        }
    }
}

/// One failed attempt, and whether trying again can help.
struct Failure {
    error: DatasetError,
    transient: bool,
}

impl Failure {
    fn transient(error: DatasetError) -> Self {
        Self {
            error,
            transient: true,
        }
    }

    fn permanent(error: DatasetError) -> Self {
        Self {
            error,
            transient: false,
        }
    }
}

/// One attempt: request `url` and stream the body into `staged`.
fn fetch(agent: &ureq::Agent, url: &str, staged: &Path) -> Result<(), Failure> {
    let response = agent.get(url).call().map_err(|e| {
        // A 4xx other than "slow down" will answer the same way next time.
        let transient = match &e {
            ureq::Error::StatusCode(code) => matches!(*code, 408 | 429 | 500..=599),
            // A malformed URL or request will fail the same way again.
            ureq::Error::BadUri(_) | ureq::Error::Http(_) => false,
            // Timeouts, resets, DNS and TLS failures can clear up.
            _ => true,
        };
        let error = DatasetError::Request {
            url: url.to_string(),
            source: Box::new(e),
        };
        Failure { error, transient }
    })?;

    let promised = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let mut file =
        fs::File::create(staged).map_err(|e| Failure::permanent(DatasetError::write(staged, e)))?;

    let body_error = |source: io::Error| {
        Failure::transient(DatasetError::ResponseBody {
            url: url.to_string(),
            source,
        })
    };

    let mut reader = response.into_body().into_reader();
    let mut chunk = vec![0u8; CHUNK];
    let mut received: u64 = 0;
    loop {
        let n = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(body_error(e)),
        };
        file.write_all(&chunk[..n])
            .map_err(|e| Failure::permanent(DatasetError::write(staged, e)))?;
        received = received.saturating_add(u64::try_from(n).unwrap_or(u64::MAX));
    }

    if let Some(promised) = promised
        && promised != received
    {
        return Err(body_error(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("received {received} of {promised} bytes"),
        )));
    }
    file.flush()
        .map_err(|e| Failure::permanent(DatasetError::write(staged, e)))
}

/// Move the finished download onto its final path.
fn publish(staged: &Path, dest: &Path) -> DatasetResult<()> {
    fs::rename(staged, dest).map_err(|e| {
        remove_staged(staged);
        DatasetError::write(dest, e).log()
    })
}

/// Drop a partial download; a missing file is the normal case after a clean failure.
fn remove_staged(staged: &Path) {
    if let Err(error) = fs::remove_file(staged)
        && error.kind() != io::ErrorKind::NotFound
    {
        warn!(path = %staged.display(), %error, "could not remove a partial download");
    }
}

/// Where the body is staged before it is renamed onto `dest`.
fn staging_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(OsString::from).unwrap_or_default();
    name.push(".part");
    dest.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Instant;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// What the test server does with one incoming connection.
    enum Step {
        /// A complete `200` response.
        Body(Vec<u8>),
        /// An empty response with this status.
        Status(u16),
        /// Promises `claimed` bytes, sends `sent`, then closes the connection.
        Truncated { sent: usize, claimed: usize },
        /// Promises `claimed` bytes, sends `sent`, then goes silent without closing.
        Stall { sent: usize, claimed: usize },
    }

    struct TestServer {
        url: String,
        connections: Arc<AtomicUsize>,
    }

    impl TestServer {
        fn connections(&self) -> usize {
            self.connections.load(Ordering::SeqCst)
        }
    }

    /// Serve one scripted [`Step`] per connection, in order, on a loopback port.
    fn serve(script: Vec<Step>) -> Result<TestServer, io::Error> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://{}/data.bin", listener.local_addr()?);
        let connections = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&connections);

        std::thread::spawn(move || {
            for step in script {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                counter.fetch_add(1, Ordering::SeqCst);
                let mut request = [0u8; 2048];
                let _ = stream.read(&mut request);

                let head = |status: &str, length: usize| {
                    format!(
                        "HTTP/1.1 {status}\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n"
                    )
                };
                match step {
                    Step::Body(body) => {
                        let _ = stream.write_all(head("200 OK", body.len()).as_bytes());
                        let _ = stream.write_all(&body);
                    }
                    Step::Status(code) => {
                        let _ = stream.write_all(head(&format!("{code} Error"), 0).as_bytes());
                    }
                    Step::Truncated { sent, claimed } => {
                        let _ = stream.write_all(head("200 OK", claimed).as_bytes());
                        let _ = stream.write_all(&vec![b'x'; sent]);
                    }
                    Step::Stall { sent, claimed } => {
                        let _ = stream.write_all(head("200 OK", claimed).as_bytes());
                        let _ = stream.write_all(&vec![b'x'; sent]);
                        let _ = stream.flush();
                        std::thread::sleep(Duration::from_secs(5));
                    }
                }
            }
        });

        Ok(TestServer { url, connections })
    }

    /// Short limits so the failure paths finish in a fraction of a second.
    fn quick(attempts: u32) -> Policy {
        Policy {
            connect: Duration::from_secs(1),
            read: Duration::from_millis(300),
            attempts,
            backoff: Duration::from_millis(10),
        }
    }

    /// Run a download on a worker thread and fail instead of hanging if it does not return.
    fn download_within(
        deadline: Duration,
        url: &str,
        dest: &Path,
        policy: Policy,
    ) -> Result<DatasetResult<()>, mpsc::RecvTimeoutError> {
        let (tx, rx) = mpsc::channel();
        let (url, dest) = (url.to_string(), dest.to_path_buf());
        std::thread::spawn(move || {
            let _ = tx.send(download(&url, &dest, &policy));
        });
        rx.recv_timeout(deadline)
    }

    #[test]
    fn completes_a_download_and_leaves_no_staging_file() -> TestResult {
        let dir = tempfile::tempdir()?;
        let dest = dir.path().join("nested").join("data.bin");
        let body = (0..200_000u32)
            .map(|i| (i % 251) as u8)
            .collect::<Vec<u8>>();
        let server = serve(vec![Step::Body(body.clone())])?;

        download(&server.url, &dest, &quick(1))?;

        assert_eq!(fs::read(&dest)?, body);
        assert!(
            !staging_path(&dest).exists(),
            "the staging file must be renamed away"
        );
        Ok(())
    }

    #[test]
    fn abandons_a_stalled_transfer_instead_of_blocking() -> TestResult {
        let dir = tempfile::tempdir()?;
        let dest = dir.path().join("data.bin");
        let server = serve(vec![Step::Stall {
            sent: 10,
            claimed: 1000,
        }])?;

        let started = Instant::now();
        let outcome = download_within(Duration::from_secs(3), &server.url, &dest, quick(1))?;

        assert!(
            matches!(outcome, Err(DatasetError::ResponseBody { .. })),
            "a stalled body must surface as a response-body error, got {outcome:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(3));
        assert!(!dest.exists(), "nothing may reach the final path");
        assert!(
            !staging_path(&dest).exists(),
            "the staging file must be removed"
        );
        Ok(())
    }

    #[test]
    fn rejects_a_truncated_body_and_keeps_an_existing_file() -> TestResult {
        let dir = tempfile::tempdir()?;
        let dest = dir.path().join("data.bin");
        fs::write(&dest, b"previous contents")?;
        let server = serve(vec![Step::Truncated {
            sent: 10,
            claimed: 1000,
        }])?;

        let outcome = download(&server.url, &dest, &quick(1));

        assert!(matches!(outcome, Err(DatasetError::ResponseBody { .. })));
        assert_eq!(fs::read(&dest)?, b"previous contents");
        assert!(!staging_path(&dest).exists());
        Ok(())
    }

    #[test]
    fn retries_a_transient_server_error() -> TestResult {
        let dir = tempfile::tempdir()?;
        let dest = dir.path().join("data.bin");
        let server = serve(vec![Step::Status(503), Step::Body(b"payload".to_vec())])?;

        download(&server.url, &dest, &quick(2))?;

        assert_eq!(fs::read(&dest)?, b"payload");
        assert_eq!(server.connections(), 2);
        Ok(())
    }

    #[test]
    fn retries_a_truncated_transfer() -> TestResult {
        let dir = tempfile::tempdir()?;
        let dest = dir.path().join("data.bin");
        let server = serve(vec![
            Step::Truncated {
                sent: 3,
                claimed: 7,
            },
            Step::Body(b"payload".to_vec()),
        ])?;

        download(&server.url, &dest, &quick(2))?;

        assert_eq!(fs::read(&dest)?, b"payload");
        assert_eq!(server.connections(), 2);
        Ok(())
    }

    #[test]
    fn does_not_retry_a_missing_file() -> TestResult {
        let dir = tempfile::tempdir()?;
        let dest = dir.path().join("data.bin");
        let server = serve(vec![Step::Status(404), Step::Body(b"payload".to_vec())])?;

        let outcome = download(&server.url, &dest, &quick(3));

        assert!(matches!(outcome, Err(DatasetError::Request { .. })));
        assert_eq!(
            server.connections(),
            1,
            "a 404 is permanent, retrying it only wastes time"
        );
        assert!(!dest.exists());
        Ok(())
    }

    #[test]
    fn gives_up_after_the_configured_attempts() -> TestResult {
        let dir = tempfile::tempdir()?;
        let dest = dir.path().join("data.bin");
        let server = serve(vec![
            Step::Status(503),
            Step::Status(503),
            Step::Status(503),
        ])?;

        let outcome = download(&server.url, &dest, &quick(3));

        assert!(matches!(outcome, Err(DatasetError::Request { .. })));
        assert_eq!(server.connections(), 3);
        assert!(!dest.exists());
        Ok(())
    }
}
