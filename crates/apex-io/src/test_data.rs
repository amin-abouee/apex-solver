//! Utilities for ensuring test datasets are available, downloading them on demand.
//!
//! Call [`ensure_odometry_dataset`] or [`ensure_ba_dataset`] at the start of a test
//! to guarantee the required file is present before the test logic runs.
//!
//! # Example
//!
//! ```no_run
//! use apex_io::test_data::ensure_odometry_dataset;
//!
//! // In a test: download intel.g2o if it isn't already there
//! let path = ensure_odometry_dataset("intel").expect("failed to fetch intel dataset");
//! ```

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::{BUNDLE_ADJUSTMENT_DATA_DIR, ODOMETRY_DATA_DIR};

// ---------------------------------------------------------------------------
// Odometry dataset registry
// ---------------------------------------------------------------------------

struct OdometryEntry {
    url: &'static str,
}

fn odometry_registry(name: &str) -> Option<OdometryEntry> {
    match name {
        "intel" => Some(OdometryEntry {
            url: "https://www.dropbox.com/s/vcz8cag7bo0zlaj/input_INTEL_g2o.g2o?dl=1",
        }),
        "mit" => Some(OdometryEntry {
            url: "https://www.dropbox.com/s/d8fcn1jg1mebx8f/input_MITb_g2o.g2o?dl=1",
        }),
        "M3500" => Some(OdometryEntry {
            url: "https://www.dropbox.com/s/gmdzo74b3tzvbrw/input_M3500_g2o.g2o?dl=1",
        }),
        "parking-garage" => Some(OdometryEntry {
            url: "https://www.dropbox.com/s/zu23p8d522qccor/parking-garage.g2o?dl=1",
        }),
        "sphere2500" => Some(OdometryEntry {
            url: "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/sphere2500.g2o",
        }),
        "torus3D" => Some(OdometryEntry {
            url: "https://www.dropbox.com/s/o95o2lbvww1100r/torus3D.g2o?dl=1",
        }),
        "cubicle" => Some(OdometryEntry {
            url: "https://www.dropbox.com/s/twpqdfphdw4md94/cubicle.g2o?dl=1",
        }),
        "rim" => Some(OdometryEntry {
            url: "https://www.dropbox.com/s/25qijwvfpmzh257/rim.g2o?dl=1",
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Ensure an odometry `.g2o` dataset is present at `data/odometry/{name}.g2o`.
///
/// If the file already exists it is returned immediately. Otherwise it is
/// downloaded from the known URL registry and saved to disk.
///
/// # Errors
/// Returns an `io::Error` if the dataset is not in the registry, the download
/// fails, or the file cannot be written.
pub fn ensure_odometry_dataset(name: &str) -> io::Result<PathBuf> {
    let path = PathBuf::from(format!("{}/{}.g2o", ODOMETRY_DATA_DIR, name));
    if path.exists() {
        return Ok(path);
    }

    let entry = odometry_registry(name).ok_or_else(|| {
        io::Error::other(format!(
            "No download URL registered for odometry dataset '{name}'"
        ))
    })?;

    eprintln!("Downloading {name}.g2o ...");
    download_file(entry.url, &path)?;
    eprintln!("Saved to {}", path.display());
    Ok(path)
}

/// Ensure a BAL bundle-adjustment dataset is present at
/// `data/bundle_adjustment/{name}/problem-{cameras}-{points}-pre.txt`.
///
/// If the file already exists it is returned immediately. Otherwise the
/// compressed `.bz2` file is downloaded from the BAL server, decompressed,
/// and saved to disk.
///
/// # Errors
/// Returns an `io::Error` if the download fails, decompression fails, or the
/// file cannot be written.
pub fn ensure_ba_dataset(name: &str, cameras: u32, points: u32) -> io::Result<PathBuf> {
    let txt_path = PathBuf::from(format!(
        "{}/{}/problem-{}-{}-pre.txt",
        BUNDLE_ADJUSTMENT_DATA_DIR, name, cameras, points
    ));
    if txt_path.exists() {
        return Ok(txt_path);
    }

    // BAL URL pattern: https://grail.cs.washington.edu/projects/bal/data/{name}/problem-{c}-{p}-pre.txt.bz2
    let url = format!(
        "https://grail.cs.washington.edu/projects/bal/data/{name}/problem-{cameras}-{points}-pre.txt.bz2"
    );

    eprintln!("Downloading {name}/problem-{cameras}-{points} ...");

    // Download to a temporary .bz2 file next to the target
    let bz2_path = txt_path.with_extension("txt.bz2");
    download_file(&url, &bz2_path)?;

    // Decompress
    decompress_bzip2(&bz2_path, &txt_path)?;
    let _ = fs::remove_file(&bz2_path); // clean up compressed file, ignore errors

    eprintln!("Saved to {}", txt_path.display());
    Ok(txt_path)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn download_file(url: &str, dest: &Path) -> io::Result<()> {
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

fn decompress_bzip2(src: &Path, dest: &Path) -> io::Result<()> {
    use bzip2::read::BzDecoder;

    let compressed = fs::File::open(src)?;
    let mut decoder = BzDecoder::new(compressed);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;

    let mut out = fs::File::create(dest)?;
    out.write_all(&decompressed)?;
    Ok(())
}
