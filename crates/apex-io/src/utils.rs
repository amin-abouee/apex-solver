//! Dataset utilities — registry, download helpers, and on-demand ensure functions.
//!
//! All dataset metadata (names, URLs, categories) lives in `datasets.toml`, which is
//! embedded at compile time. No URLs are hardcoded in Rust source.
//!
//! # Usage in tests
//!
//! ```no_run
//! use apex_io::ensure_odometry_dataset;
//!
//! let path = ensure_odometry_dataset("sphere2500").expect("failed to fetch dataset");
//! // path == "data/odometry/3d/sphere2500.g2o"
//! ```
//!
//! # Usage in the download binary
//!
//! ```no_run
//! use apex_io::utils::DatasetRegistry;
//!
//! let registry = DatasetRegistry::load().unwrap();
//! for (name, entry) in registry.odometry_by_category("3d") {
//!     println!("{name}: {}", entry.url);
//! }
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::info;

use crate::{BUNDLE_ADJUSTMENT_DATA_DIR, ODOMETRY_DATA_DIR};

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

/// The complete dataset registry, parsed from `datasets.toml`.
#[derive(Debug, Deserialize)]
pub struct DatasetRegistry {
    /// Odometry datasets keyed by short name (e.g. `"sphere2500"`, `"intel"`).
    pub odometry: HashMap<String, OdometryEntry>,
    /// Bundle adjustment datasets keyed by collection name (e.g. `"ladybug"`).
    pub bundle_adjustment: HashMap<String, BaEntry>,
}

impl DatasetRegistry {
    /// Load the registry from the compile-time embedded `datasets.toml`.
    ///
    /// # Errors
    /// Returns an error only if `datasets.toml` is malformed TOML — a
    /// developer error that should never occur with the bundled file.
    pub fn load() -> io::Result<Self> {
        toml::from_str(DATASETS_TOML).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Returns the on-disk path for an odometry dataset, including its category subdirectory.
    ///
    /// Returns `None` if `name` is not in the registry.
    ///
    /// # Example
    /// ```
    /// use apex_io::DatasetRegistry;
    /// # fn main() -> std::io::Result<()> {
    /// let reg = DatasetRegistry::load()?;
    /// assert_eq!(
    ///     reg.odometry_path("intel").map(|p| p.to_str().unwrap().to_string()),
    ///     Some("data/odometry/2d/intel.g2o".to_string())
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
pub fn ensure_odometry_dataset(name: &str) -> io::Result<PathBuf> {
    let registry = DatasetRegistry::load()?;

    let entry = registry.odometry.get(name).ok_or_else(|| {
        io::Error::other(format!(
            "Dataset '{name}' not found in registry. \
             Available: {}",
            {
                let mut names: Vec<_> = registry.odometry.keys().map(String::as_str).collect();
                names.sort();
                names.join(", ")
            }
        ))
    })?;

    let path = PathBuf::from(ODOMETRY_DATA_DIR)
        .join(&entry.category)
        .join(&entry.filename);
    if path.exists() {
        return Ok(path);
    }

    info!("Downloading {name} ({}) ...", entry.filename);
    download_file(&entry.url, &path)
        .map_err(|e| io::Error::other(format!("Failed to download {name}: {e}")))?;
    info!("Saved to {}", path.display());
    Ok(path)
}

/// Ensure a BAL bundle-adjustment file is present at
/// `data/bundle_adjustment/{name}/problem-{cameras}-{points}-pre.txt`.
///
/// If the file already exists it is returned immediately. Otherwise the
/// `.bz2` archive is downloaded, decompressed, and the `.bz2` is cleaned up.
///
/// # Errors
/// Returns an error if the download, decompression, or disk write fails.
pub fn ensure_ba_dataset(name: &str, cameras: u32, points: u32) -> io::Result<PathBuf> {
    let txt_path = PathBuf::from(BUNDLE_ADJUSTMENT_DATA_DIR)
        .join(name)
        .join(format!("problem-{cameras}-{points}-pre.txt"));

    if txt_path.exists() {
        return Ok(txt_path);
    }

    let registry = DatasetRegistry::load()?;
    let entry = registry.bundle_adjustment.get(name).ok_or_else(|| {
        io::Error::other(format!(
            "BA dataset '{name}' not found in registry. \
             Available: {}",
            {
                let mut names: Vec<_> = registry
                    .bundle_adjustment
                    .keys()
                    .map(String::as_str)
                    .collect();
                names.sort();
                names.join(", ")
            }
        ))
    })?;

    let url = entry.problem_url(cameras, points);
    let bz2_path = txt_path.with_extension("txt.bz2");

    info!("Downloading {name}/problem-{cameras}-{points} ...");
    download_file(&url, &bz2_path)
        .map_err(|e| io::Error::other(format!("Failed to download {name}: {e}")))?;

    decompress_bzip2(&bz2_path, &txt_path)
        .map_err(|e| io::Error::other(format!("Failed to decompress: {e}")))?;

    let _ = fs::remove_file(&bz2_path); // clean up; ignore errors
    info!("Saved to {}", txt_path.display());
    Ok(txt_path)
}

// ---------------------------------------------------------------------------
// Low-level download helpers (pub so the download_datasets binary can use them)
// ---------------------------------------------------------------------------

/// Download a URL to a local file, creating parent directories as needed.
///
/// # Errors
/// Returns an error if the HTTP request fails or the file cannot be written.
pub fn download_file(url: &str, dest: &Path) -> io::Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let response = ureq::get(url)
        .call()
        .map_err(|e| io::Error::other(format!("HTTP request failed for {url}: {e}")))?;

    let mut buf = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| io::Error::other(format!("Failed to read response body: {e}")))?;

    let mut file = fs::File::create(dest)?;
    file.write_all(&buf)?;
    Ok(())
}

/// Decompress a `.bz2` file to `dest`.
///
/// # Errors
/// Returns an error if the file cannot be read or the decompressed data
/// cannot be written.
pub fn decompress_bzip2(src: &Path, dest: &Path) -> io::Result<()> {
    use bzip2::read::BzDecoder;

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let compressed = fs::File::open(src)?;
    let mut decoder = BzDecoder::new(compressed);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;

    let mut out = fs::File::create(dest)?;
    out.write_all(&decompressed)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_parses_without_panic() -> io::Result<()> {
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
    fn registry_contains_expected_odometry_datasets() -> io::Result<()> {
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
    fn registry_contains_expected_ba_datasets() -> io::Result<()> {
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
    fn odometry_entries_have_valid_categories() -> io::Result<()> {
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
    fn ba_entries_have_at_least_one_problem() -> io::Result<()> {
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
    fn ba_problem_url_format_is_correct() -> io::Result<()> {
        let registry = DatasetRegistry::load()?;
        let ladybug = registry
            .bundle_adjustment
            .get("ladybug")
            .ok_or_else(|| io::Error::other("ladybug dataset not found"))?;
        let url = ladybug.problem_url(49, 7776);
        assert_eq!(
            url,
            "https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2"
        );
        Ok(())
    }

    #[test]
    fn odometry_by_category_returns_only_3d() -> io::Result<()> {
        let registry = DatasetRegistry::load()?;
        let entries = registry.odometry_by_category("3d");
        for (_, entry) in &entries {
            assert_eq!(entry.category, "3d");
        }
        assert!(!entries.is_empty());
        Ok(())
    }

    #[test]
    fn sphere2500_uses_github_url() -> io::Result<()> {
        let registry = DatasetRegistry::load()?;
        let entry = registry
            .odometry
            .get("sphere2500")
            .ok_or_else(|| io::Error::other("sphere2500 must exist"))?;
        assert!(
            entry.url.contains("github"),
            "sphere2500 should use the GitHub URL, got: {}",
            entry.url
        );
        Ok(())
    }

    #[test]
    fn registry_contains_new_vertigo_datasets() -> io::Result<()> {
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
    fn odometry_path_includes_category_subdir() -> io::Result<()> {
        let registry = DatasetRegistry::load()?;
        let path_3d = registry
            .odometry_path("sphere2500")
            .ok_or_else(|| io::Error::other("sphere2500 path not found"))?;
        let path_2d = registry
            .odometry_path("intel")
            .ok_or_else(|| io::Error::other("intel path not found"))?;
        assert!(
            path_3d.to_str().is_some_and(|s| s.contains("/3d/")),
            "3D path should contain /3d/, got: {}",
            path_3d.display()
        );
        assert!(
            path_2d.to_str().is_some_and(|s| s.contains("/2d/")),
            "2D path should contain /2d/, got: {}",
            path_2d.display()
        );
        Ok(())
    }

    #[test]
    fn sphere_bignoise_removed_from_registry() -> io::Result<()> {
        let registry = DatasetRegistry::load()?;
        assert!(
            !registry.odometry.contains_key("sphere_bignoise"),
            "sphere_bignoise should have been removed (merged into sphere2500)"
        );
        Ok(())
    }
}
