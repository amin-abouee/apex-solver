//! Dataset utilities — registry, download helpers, and on-demand ensure functions.
//!
//! All dataset metadata (names, URLs, categories) lives in `datasets.toml`, which is
//! embedded at compile time. No URLs are hardcoded in Rust source.
//!
//! # Usage in tests
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use apex_io::ensure_odometry_dataset;
//!
//! let path = ensure_odometry_dataset("sphere2500")?;
//! // path == "data/odometry/3d/sphere2500.g2o"
//! # Ok(())
//! # }
//! ```
//!
//! # Usage in the download binary
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use apex_io::utils::DatasetRegistry;
//!
//! let registry = DatasetRegistry::load()?;
//! for (name, entry) in registry.odometry_by_category("3d") {
//!     println!("{name}: {}", entry.url);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
#[cfg(feature = "download")]
use std::fs;
use std::io;
#[cfg(feature = "download")]
use std::io::{Read, Write};
#[cfg(feature = "download")]
use std::path::Path;
use std::path::PathBuf;

use crate::error::ErrorLogging;
use serde::Deserialize;
#[cfg(feature = "download")]
use tracing::{info, warn};

use crate::{BUNDLE_ADJUSTMENT_DATA_DIR, ODOMETRY_DATA_DIR, SENSOR_DATA_DIR};

// Compile-time embed of the dataset registry.
const DATASETS_TOML: &str = include_str!("../datasets.toml");

// ---------------------------------------------------------------------------
// Registry types
// ---------------------------------------------------------------------------

/// Metadata for a single odometry (pose graph) dataset.
#[derive(Debug, Clone, Deserialize)]
pub struct OdometryEntry {
    /// Direct download URL for the `.g2o` file.
    pub url: String,
    /// Filename on disk (saved to `data/odometry/<filename>`).
    pub filename: String,
    /// Pose graph dimensionality: `"2d"` or `"3d"`.
    pub category: String,
}

/// Metadata for a bundle adjustment (BAL) dataset collection.
#[derive(Debug, Clone, Deserialize)]
pub struct BaEntry {
    /// URL prefix; full URL = `{url_prefix}/problem-{cameras}-{points}-pre.txt.bz2`.
    pub url_prefix: String,
    /// All available (cameras, points) problem sizes in this collection.
    pub problems: Vec<[u32; 2]>,
}

impl BaEntry {
    /// Returns the largest problem (most cameras) in this collection.
    pub fn largest(&self) -> Option<[u32; 2]> {
        self.problems.last().copied()
    }

    /// Constructs the download URL for a specific problem size.
    pub fn problem_url(&self, cameras: u32, points: u32) -> String {
        format!(
            "{}/problem-{}-{}-pre.txt.bz2",
            self.url_prefix, cameras, points
        )
    }
}

/// Metadata for a multi-sensor dataset (IMU / GNSS / odometry with ground truth).
///
/// Plain files are downloaded as-is; archives are downloaded and unpacked in
/// place. Both are `(url, filename)` pairs saved under
/// `data/sensor/<name>/`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct SensorEntry {
    /// `(url, filename)` pairs downloaded verbatim.
    #[serde(default)]
    pub files: Vec<(String, String)>,
    /// `(url, filename)` `.tar.gz` archives, unpacked into the dataset directory.
    #[serde(default)]
    pub archives: Vec<(String, String)>,
}

/// The complete dataset registry, parsed from `datasets.toml`.
#[derive(Debug, Deserialize)]
pub struct DatasetRegistry {
    /// Odometry datasets keyed by short name (e.g. `"sphere2500"`, `"intel"`).
    pub odometry: HashMap<String, OdometryEntry>,
    /// Bundle adjustment datasets keyed by collection name (e.g. `"ladybug"`).
    pub bundle_adjustment: HashMap<String, BaEntry>,
    /// Multi-sensor datasets keyed by short name (e.g. `"nclt-2013-01-10"`).
    #[serde(default)]
    pub sensor: HashMap<String, SensorEntry>,
}

impl DatasetRegistry {
    /// Load the registry from the compile-time embedded `datasets.toml`.
    ///
    /// # Errors
    /// Returns an error only if `datasets.toml` is malformed TOML — a
    /// developer error that should never occur with the bundled file.
    pub fn load() -> DatasetResult<Self> {
        toml::from_str(DATASETS_TOML).map_err(|e| DatasetError::Registry(e).log())
    }

    /// Returns the on-disk path for an odometry dataset, including its category subdirectory.
    ///
    /// Returns `None` if `name` is not in the registry.
    ///
    /// # Example
    /// ```
    /// use apex_io::DatasetRegistry;
    /// # fn main() -> Result<(), apex_io::utils::DatasetError> {
    /// let reg = DatasetRegistry::load()?;
    /// assert_eq!(
    ///     reg.odometry_path("intel"),
    ///     Some(std::path::Path::new("data/odometry").join("2d").join("intel.g2o"))
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn odometry_path(&self, name: &str) -> Option<std::path::PathBuf> {
        self.odometry.get(name).map(|e| {
            std::path::PathBuf::from(crate::ODOMETRY_DATA_DIR)
                .join(&e.category)
                .join(&e.filename)
        })
    }

    /// Returns all odometry entries with the given category (`"2d"` or `"3d"`),
    /// sorted alphabetically by name for deterministic output.
    pub fn odometry_by_category(&self, category: &str) -> Vec<(&str, &OdometryEntry)> {
        let mut entries: Vec<_> = self
            .odometry
            .iter()
            .filter(|(_, e)| e.category == category)
            .map(|(name, entry)| (name.as_str(), entry))
            .collect();
        entries.sort_by_key(|(name, _)| *name);
        entries
    }

    /// Returns the on-disk path for a specific BA problem file.
    ///
    /// The path follows the same layout the downloader creates:
    /// `data/bundle_adjustment/{name}/problem-{cameras}-{points}-pre.txt`
    ///
    /// Returns `None` if `name` is not in the registry.
    pub fn ba_path(&self, name: &str, cameras: u32, points: u32) -> Option<std::path::PathBuf> {
        self.bundle_adjustment.get(name).map(|_| {
            std::path::PathBuf::from(crate::BUNDLE_ADJUSTMENT_DATA_DIR)
                .join(name)
                .join(format!("problem-{cameras}-{points}-pre.txt"))
        })
    }

    /// Returns all bundle adjustment entries sorted alphabetically by name.
    pub fn ba_sorted(&self) -> Vec<(&str, &BaEntry)> {
        let mut entries: Vec<_> = self
            .bundle_adjustment
            .iter()
            .map(|(name, entry)| (name.as_str(), entry))
            .collect();
        entries.sort_by_key(|(name, _)| *name);
        entries
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Result of a dataset registry lookup or download.
pub type DatasetResult<T> = Result<T, DatasetError>;

/// A dataset that cannot be found, downloaded or unpacked.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DatasetError {
    #[error("the embedded dataset registry is not valid TOML")]
    Registry(#[source] toml::de::Error),

    #[error("{kind} dataset {name} is not in the registry, available: {}", available.join(", "))]
    UnknownDataset {
        kind: &'static str,
        name: String,
        available: Vec<String>,
    },

    #[error(
        "dataset {name} is not present locally and apex-io was built without the download feature"
    )]
    DownloadDisabled { name: String },

    #[cfg(feature = "download")]
    #[error("HTTP request for {url} failed")]
    Request {
        url: String,
        #[source]
        source: Box<ureq::Error>,
    },

    #[error("failed to read the response body of {url}")]
    ResponseBody {
        url: String,
        #[source]
        source: io::Error,
    },

    #[error("failed to write {}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to decompress {}", path.display())]
    Decompress {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to extract {}", path.display())]
    Extract {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

impl DatasetError {
    fn unknown<'a>(
        kind: &'static str,
        name: &str,
        available: impl Iterator<Item = &'a String>,
    ) -> Self {
        let mut available: Vec<String> = available.cloned().collect();
        available.sort();
        Self::UnknownDataset {
            kind,
            name: name.to_string(),
            available,
        }
    }

    #[cfg(feature = "download")]
    fn write(path: &Path, source: io::Error) -> Self {
        Self::Write {
            path: path.to_path_buf(),
            source,
        }
    }
}

/// Remove a downloaded archive once unpacked; a leftover file is only clutter.
#[cfg(feature = "download")]
fn remove_archive(path: &Path) {
    if let Err(error) = fs::remove_file(path) {
        warn!(path = %path.display(), %error, "could not remove a downloaded archive");
    }
}

// ---------------------------------------------------------------------------
// Public ensure API (used by tests and binaries)
// ---------------------------------------------------------------------------

/// Ensure an odometry `.g2o` dataset is present at `data/odometry/{name}.g2o`.
///
/// If the file already exists it is returned immediately (no network access).
/// Otherwise it is looked up in the dataset registry and downloaded.
///
/// # Errors
/// Returns an error if the dataset name is not in the registry, the download
/// fails, or the file cannot be written.
pub fn ensure_odometry_dataset(name: &str) -> DatasetResult<PathBuf> {
    let registry = DatasetRegistry::load()?;

    let entry = registry
        .odometry
        .get(name)
        .ok_or_else(|| DatasetError::unknown("odometry", name, registry.odometry.keys()).log())?;

    let path = PathBuf::from(ODOMETRY_DATA_DIR)
        .join(&entry.category)
        .join(&entry.filename);
    if path.exists() {
        return Ok(path);
    }

    #[cfg(feature = "download")]
    {
        info!("Downloading {name} ({}) ...", entry.filename);
        download_file(&entry.url, &path)?;
        info!("Saved to {}", path.display());
        Ok(path)
    }
    #[cfg(not(feature = "download"))]
    {
        Err(DatasetError::DownloadDisabled {
            name: name.to_string(),
        }
        .log())
    }
}

/// Ensure a BAL bundle-adjustment file is present at
/// `data/bundle_adjustment/{name}/problem-{cameras}-{points}-pre.txt`.
///
/// If the file already exists it is returned immediately. Otherwise the
/// `.bz2` archive is downloaded, decompressed, and the `.bz2` is cleaned up.
///
/// # Errors
/// Returns an error if the download, decompression, or disk write fails.
pub fn ensure_ba_dataset(name: &str, cameras: u32, points: u32) -> DatasetResult<PathBuf> {
    let txt_path = PathBuf::from(BUNDLE_ADJUSTMENT_DATA_DIR)
        .join(name)
        .join(format!("problem-{cameras}-{points}-pre.txt"));

    if txt_path.exists() {
        return Ok(txt_path);
    }

    let registry = DatasetRegistry::load()?;
    let entry = registry.bundle_adjustment.get(name).ok_or_else(|| {
        DatasetError::unknown("bundle adjustment", name, registry.bundle_adjustment.keys()).log()
    })?;

    #[cfg(feature = "download")]
    {
        let url = entry.problem_url(cameras, points);
        let bz2_path = txt_path.with_extension("txt.bz2");
        info!("Downloading {name}/problem-{cameras}-{points} ...");
        download_file(&url, &bz2_path)?;
        decompress_bzip2(&bz2_path, &txt_path)?;
        remove_archive(&bz2_path);
        info!("Saved to {}", txt_path.display());
        Ok(txt_path)
    }
    #[cfg(not(feature = "download"))]
    {
        let _ = entry;
        Err(DatasetError::DownloadDisabled {
            name: format!("{name}/problem-{cameras}-{points}"),
        }
        .log())
    }
}

/// Ensure a multi-sensor dataset is present under `data/sensor/{name}/`.
///
/// Each declared file is downloaded once; each declared `.tar.gz` archive is
/// downloaded, unpacked in place, and the archive removed. A dataset whose
/// marker file already exists is returned immediately, with no network access.
///
/// Unlike the image and LIDAR streams these datasets also publish, the CSV
/// streams fetched here are a few tens of megabytes.
///
/// # Errors
/// Returns an error if the name is not in the registry, or if a download or
/// extraction fails.
pub fn ensure_sensor_dataset(name: &str) -> DatasetResult<PathBuf> {
    let registry = DatasetRegistry::load()?;
    let entry = registry
        .sensor
        .get(name)
        .ok_or_else(|| DatasetError::unknown("sensor", name, registry.sensor.keys()).log())?;

    let dir = PathBuf::from(SENSOR_DATA_DIR).join(name);
    // The extraction marker: present only after every archive has been unpacked.
    let marker = dir.join(".complete");
    if marker.exists() {
        return Ok(dir);
    }

    #[cfg(feature = "download")]
    {
        fs::create_dir_all(&dir).map_err(|e| DatasetError::write(&dir, e).log())?;

        for (url, filename) in &entry.files {
            let dest = dir.join(filename);
            if dest.exists() {
                continue;
            }
            info!("Downloading {name}/{filename} ...");
            download_file(url, &dest)?;
        }

        for (url, filename) in &entry.archives {
            let archive = dir.join(filename);
            info!("Downloading {name}/{filename} ...");
            download_file(url, &archive)?;
            extract_tar_gz(&archive, &dir)?;
            remove_archive(&archive);
        }

        fs::write(&marker, b"").map_err(|e| DatasetError::write(&marker, e).log())?;
        info!("Sensor dataset ready at {}", dir.display());
        Ok(dir)
    }
    #[cfg(not(feature = "download"))]
    {
        let _ = entry;
        Err(DatasetError::DownloadDisabled {
            name: name.to_string(),
        }
        .log())
    }
}

// ---------------------------------------------------------------------------
// Low-level download helpers (pub so the download_datasets binary can use them)
// ---------------------------------------------------------------------------

/// Download a URL to a local file, creating parent directories as needed.
///
/// Requires the `download` feature.
///
/// # Errors
/// Returns an error if the HTTP request fails or the file cannot be written.
#[cfg(feature = "download")]
pub fn download_file(url: &str, dest: &Path) -> DatasetResult<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| DatasetError::write(parent, e).log())?;
    }

    let response = ureq::get(url).call().map_err(|e| {
        DatasetError::Request {
            url: url.to_string(),
            source: Box::new(e),
        }
        .log()
    })?;

    let mut buf = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut buf)
        .map_err(|source| {
            DatasetError::ResponseBody {
                url: url.to_string(),
                source,
            }
            .log()
        })?;

    fs::File::create(dest)
        .and_then(|mut file| file.write_all(&buf))
        .map_err(|e| DatasetError::write(dest, e).log())
}

/// Unpack a `.tar.gz` archive into `dest_dir`.
///
/// Requires the `download` feature.
///
/// # Errors
/// Returns an error if the archive cannot be read or an entry cannot be written.
#[cfg(feature = "download")]
pub fn extract_tar_gz(src: &Path, dest_dir: &Path) -> DatasetResult<()> {
    fs::File::open(src)
        .and_then(|file| tar::Archive::new(flate2::read::GzDecoder::new(file)).unpack(dest_dir))
        .map_err(|source| {
            DatasetError::Extract {
                path: src.to_path_buf(),
                source,
            }
            .log()
        })
}

/// Decompress a `.bz2` file to `dest`.
///
/// Requires the `download` feature.
///
/// # Errors
/// Returns an error if the file cannot be read or the decompressed data
/// cannot be written.
#[cfg(feature = "download")]
pub fn decompress_bzip2(src: &Path, dest: &Path) -> DatasetResult<()> {
    use bzip2::read::BzDecoder;

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| DatasetError::write(parent, e).log())?;
    }

    let mut decompressed = Vec::new();
    fs::File::open(src)
        .and_then(|compressed| BzDecoder::new(compressed).read_to_end(&mut decompressed))
        .map_err(|source| {
            DatasetError::Decompress {
                path: src.to_path_buf(),
                source,
            }
            .log()
        })?;

    fs::File::create(dest)
        .and_then(|mut out| out.write_all(&decompressed))
        .map_err(|e| DatasetError::write(dest, e).log())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_parses_without_panic() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        assert!(
            !registry.odometry.is_empty(),
            "odometry section must not be empty"
        );
        assert!(
            !registry.bundle_adjustment.is_empty(),
            "bundle_adjustment section must not be empty"
        );
        Ok(())
    }

    #[test]
    fn registry_contains_expected_odometry_datasets() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        for name in &["sphere2500", "parking-garage", "intel", "M3500"] {
            assert!(
                registry.odometry.contains_key(*name),
                "missing expected dataset: {name}"
            );
        }
        Ok(())
    }

    #[test]
    fn registry_contains_expected_ba_datasets() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        for name in &["ladybug", "trafalgar", "dubrovnik", "venice", "final"] {
            assert!(
                registry.bundle_adjustment.contains_key(*name),
                "missing expected BA dataset: {name}"
            );
        }
        Ok(())
    }

    #[test]
    fn odometry_entries_have_valid_categories() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        for (name, entry) in &registry.odometry {
            assert!(
                entry.category == "2d" || entry.category == "3d",
                "dataset '{name}' has invalid category: '{}'",
                entry.category
            );
        }
        Ok(())
    }

    #[test]
    fn ba_entries_have_at_least_one_problem() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        for (name, entry) in &registry.bundle_adjustment {
            assert!(
                !entry.problems.is_empty(),
                "BA dataset '{name}' has no problems listed"
            );
        }
        Ok(())
    }

    #[test]
    fn ba_problem_url_format_is_correct() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let ladybug = registry
            .bundle_adjustment
            .get("ladybug")
            .ok_or_else(|| std::io::Error::other("ladybug dataset not found"))?;
        let url = ladybug.problem_url(49, 7776);
        assert_eq!(
            url,
            "https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2"
        );
        Ok(())
    }

    #[test]
    fn odometry_by_category_returns_only_3d() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let entries = registry.odometry_by_category("3d");
        for (_, entry) in &entries {
            assert_eq!(entry.category, "3d");
        }
        assert!(!entries.is_empty());
        Ok(())
    }

    #[test]
    fn sphere2500_uses_github_url() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let entry = registry
            .odometry
            .get("sphere2500")
            .ok_or_else(|| std::io::Error::other("sphere2500 must exist"))?;
        assert!(
            entry.url.contains("github"),
            "sphere2500 should use the GitHub URL, got: {}",
            entry.url
        );
        Ok(())
    }

    #[test]
    fn registry_contains_new_vertigo_datasets() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        for name in &["manhattanOlson3500", "ring", "ring_city", "city10000"] {
            assert!(
                registry.odometry.contains_key(*name),
                "missing expected dataset: {name}"
            );
        }
        Ok(())
    }

    #[test]
    fn odometry_path_includes_category_subdir() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let path_3d = registry
            .odometry_path("sphere2500")
            .ok_or_else(|| std::io::Error::other("sphere2500 path not found"))?;
        let path_2d = registry
            .odometry_path("intel")
            .ok_or_else(|| std::io::Error::other("intel path not found"))?;
        assert!(
            path_3d.components().any(|c| c.as_os_str() == "3d"),
            "3D path should contain '3d' component, got: {}",
            path_3d.display()
        );
        assert!(
            path_2d.components().any(|c| c.as_os_str() == "2d"),
            "2D path should contain '2d' component, got: {}",
            path_2d.display()
        );
        Ok(())
    }

    #[test]
    fn sphere_bignoise_removed_from_registry() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        assert!(
            !registry.odometry.contains_key("sphere_bignoise"),
            "sphere_bignoise should have been removed (merged into sphere2500)"
        );
        Ok(())
    }

    #[test]
    fn ba_path_returns_correct_structure() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let path = registry
            .ba_path("ladybug", 49, 7776)
            .ok_or_else(|| std::io::Error::other("ladybug ba_path not found"))?;
        assert!(
            path.components()
                .any(|c| c.as_os_str() == "bundle_adjustment"),
            "path should contain 'bundle_adjustment', got: {}",
            path.display()
        );
        assert!(
            path.components().any(|c| c.as_os_str() == "ladybug"),
            "path should contain 'ladybug', got: {}",
            path.display()
        );
        assert!(
            path.file_name()
                .is_some_and(|f| f == "problem-49-7776-pre.txt"),
            "filename should be 'problem-49-7776-pre.txt', got: {}",
            path.display()
        );
        Ok(())
    }

    #[test]
    fn ba_path_returns_none_for_unknown() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        assert!(
            registry.ba_path("nonexistent_ba_xyz", 1, 1).is_none(),
            "unknown BA name should return None"
        );
        Ok(())
    }

    #[test]
    fn ba_sorted_returns_alphabetical_order() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let entries = registry.ba_sorted();
        assert!(!entries.is_empty(), "ba_sorted should not be empty");
        for window in entries.windows(2) {
            assert!(
                window[0].0 <= window[1].0,
                "ba_sorted is not sorted: '{}' > '{}'",
                window[0].0,
                window[1].0
            );
        }
        Ok(())
    }

    #[test]
    fn ba_entry_largest_returns_last_problem() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let ladybug = registry
            .bundle_adjustment
            .get("ladybug")
            .ok_or_else(|| std::io::Error::other("ladybug not found"))?;
        let largest = ladybug.largest();
        assert!(largest.is_some(), "ladybug should have a largest problem");
        assert_eq!(
            largest,
            ladybug.problems.last().copied(),
            "largest() should equal the last problem"
        );
        Ok(())
    }

    #[test]
    fn ba_entry_largest_empty_returns_none() {
        let entry = BaEntry {
            url_prefix: "https://example.com".to_string(),
            problems: vec![],
        };
        assert!(
            entry.largest().is_none(),
            "empty problems should return None"
        );
    }

    #[test]
    fn odometry_by_category_returns_only_2d() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let entries = registry.odometry_by_category("2d");
        assert!(!entries.is_empty(), "should have at least one 2d dataset");
        for (_, entry) in &entries {
            assert_eq!(entry.category, "2d");
        }
        Ok(())
    }

    #[test]
    fn odometry_by_category_is_sorted() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let entries = registry.odometry_by_category("3d");
        for window in entries.windows(2) {
            assert!(
                window[0].0 <= window[1].0,
                "odometry_by_category is not sorted: '{}' > '{}'",
                window[0].0,
                window[1].0
            );
        }
        Ok(())
    }

    #[test]
    fn odometry_entries_have_nonempty_url_and_filename() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        for (name, entry) in &registry.odometry {
            assert!(!entry.url.is_empty(), "dataset '{name}' has empty url");
            assert!(
                !entry.filename.is_empty(),
                "dataset '{name}' has empty filename"
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "download")]
    fn decompress_bzip2_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        use bzip2::Compression;
        use bzip2::write::BzEncoder;
        use std::io::Write as _;

        let original = b"hello bzip2 roundtrip test data";

        // Write compressed bytes to a temp file
        let tmp_dir = tempfile::tempdir()?;
        let bz2_path = tmp_dir.path().join("test.txt.bz2");
        let txt_path = tmp_dir.path().join("test.txt");

        {
            let file = fs::File::create(&bz2_path)?;
            let mut encoder = BzEncoder::new(file, Compression::fast());
            encoder.write_all(original)?;
            encoder.finish()?;
        }

        decompress_bzip2(&bz2_path, &txt_path)?;

        let decompressed = fs::read(&txt_path)?;
        assert_eq!(
            decompressed, original,
            "decompressed content must match original"
        );
        Ok(())
    }

    #[test]
    fn ensure_odometry_dataset_unknown_name_errors() -> Result<(), Box<dyn std::error::Error>> {
        let err = ensure_odometry_dataset("nonexistent_dataset_xyz_abc")
            .err()
            .ok_or_else(|| std::io::Error::other("expected Err but got Ok"))?;
        assert!(
            matches!(err, DatasetError::UnknownDataset { .. }),
            "an unknown name must be reported as such, got: {err}"
        );
        assert!(
            err.to_string().contains("is not in the registry"),
            "error message should mention registry, got: {err}"
        );
        Ok(())
    }

    #[test]
    fn ensure_ba_dataset_unknown_name_errors() -> Result<(), Box<dyn std::error::Error>> {
        let err = ensure_ba_dataset("nonexistent_ba_xyz_abc", 1, 1)
            .err()
            .ok_or_else(|| std::io::Error::other("expected Err but got Ok"))?;
        assert!(
            matches!(err, DatasetError::UnknownDataset { .. }),
            "an unknown name must be reported as such, got: {err}"
        );
        assert!(
            err.to_string().contains("is not in the registry"),
            "error message should mention registry, got: {err}"
        );
        Ok(())
    }

    #[test]
    fn odometry_path_returns_none_for_unknown() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        assert!(registry.odometry_path("nonexistent_xyz").is_none());
        Ok(())
    }

    #[test]
    fn odometry_by_category_empty_for_unknown() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let entries = registry.odometry_by_category("4d");
        assert!(entries.is_empty());
        Ok(())
    }

    #[test]
    fn ba_problem_url_format_for_trafalgar() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let trafalgar = registry
            .bundle_adjustment
            .get("trafalgar")
            .ok_or_else(|| std::io::Error::other("trafalgar dataset not found"))?;
        let url = trafalgar.problem_url(21, 100);
        assert!(url.contains("trafalgar"));
        assert!(url.contains("problem-21-100-pre.txt.bz2"));
        Ok(())
    }

    #[test]
    fn ba_entry_largest_with_single_problem() {
        let entry = BaEntry {
            url_prefix: "https://example.com".to_string(),
            problems: vec![[10, 20]],
        };
        assert_eq!(entry.largest(), Some([10, 20]));
    }

    #[test]
    fn odometry_entry_fields_accessible() -> Result<(), Box<dyn std::error::Error>> {
        let registry = DatasetRegistry::load()?;
        let entry = registry
            .odometry
            .get("sphere2500")
            .ok_or_else(|| std::io::Error::other("sphere2500 must exist"))?;
        assert!(!entry.filename.is_empty());
        assert!(!entry.category.is_empty());
        Ok(())
    }
}
