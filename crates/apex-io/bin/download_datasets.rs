//! Dataset downloader for pose graph and bundle adjustment datasets.
//!
//! Downloads datasets from two sources:
//! 1. **Luca Carlone's pose graph datasets** (3D/2D g2o): https://lucacarlone.mit.edu/datasets/
//! 2. **UW BAL bundle adjustment datasets**: https://grail.cs.washington.edu/projects/bal/
//!
//! All dataset metadata (names, URLs, categories) lives in `datasets.toml`, which is
//! embedded at compile time. No URLs are hardcoded here.
//!
//! Usage:
//! ```bash
//! # Download specific dataset group by number
//! cargo run -p apex-io --bin download_datasets -- --select 3    # All odometry g2o
//! cargo run -p apex-io --bin download_datasets -- --select 0    # Largest from each BA
//!
//! # List available datasets
//! cargo run -p apex-io --bin download_datasets -- --list
//!
//! # Interactive mode (prompts for selection)
//! cargo run -p apex-io --bin download_datasets
//! ```

use apex_io::utils::{DatasetRegistry, decompress_bzip2, download_file};
use apex_io::{BUNDLE_ADJUSTMENT_DATA_DIR, ODOMETRY_DATA_DIR};
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Dataset selection number (see --list for options, 0-10)
    #[arg(short, long)]
    select: Option<usize>,

    /// List available datasets and exit
    #[arg(short, long)]
    list: bool,

    /// Output directory (overrides default data/ paths)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Skip confirmation prompt
    #[arg(short = 'y', long, default_value = "false")]
    yes: bool,
}

// ---------------------------------------------------------------------------
// Download helpers (wrap utils fns to track byte counts)
// ---------------------------------------------------------------------------

fn download_and_size(url: &str, dest: &PathBuf) -> Result<u64, Box<dyn std::error::Error>> {
    println!("  Downloading: {url}");
    println!("  Saving to: {dest:?}");
    download_file(url, dest).map_err(|e| format!("HTTP request failed: {e}"))?;
    let size = std::fs::metadata(dest)?.len();
    Ok(size)
}

fn decompress_and_cleanup(
    compressed: &PathBuf,
    decompressed: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Decompressing: {compressed:?}");
    println!("  Extracting to: {decompressed:?}");
    decompress_bzip2(compressed, decompressed)
        .map_err(|e| format!("Decompression failed: {e}"))?;
    if let Err(e) = std::fs::remove_file(compressed) {
        eprintln!("  Warning: could not remove compressed file: {e}");
    }
    println!("  Done!");
    Ok(())
}

// ---------------------------------------------------------------------------
// Listing
// ---------------------------------------------------------------------------

fn list_datasets(registry: &DatasetRegistry) {
    println!("\n=== Available Datasets ===\n");

    let d3: Vec<_> = registry.odometry_by_category("3d");
    let d2: Vec<_> = registry.odometry_by_category("2d");

    println!("Odometry (g2o format):");
    println!(
        "  1. 3D g2o      - {} files ({})",
        d3.len(),
        d3.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
    );
    println!(
        "  2. 2D g2o      - {} files ({})",
        d2.len(),
        d2.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
    );
    println!(
        "  3. All odometry g2o - {} files (all odometry datasets)",
        d3.len() + d2.len()
    );
    println!();

    println!("Bundle Adjustment (UW BAL format):");
    let ba = registry.ba_sorted();
    for (idx, (name, entry)) in ba.iter().enumerate() {
        let largest = entry.largest();
        let (cameras, points) = largest.map(|[c, p]| (c, p)).unwrap_or((0, 0));
        println!(
            "  {}. {:12} - {} problems (largest: {}x{})",
            idx + 4,
            name,
            entry.problems.len(),
            cameras,
            points
        );
    }

    let ba_len = ba.len();
    println!("  {}. Largest each  - {} problems (largest from each BA)", ba_len + 4, ba_len);

    let total_ba: usize = registry
        .bundle_adjustment
        .values()
        .map(|e| e.problems.len())
        .sum();
    let total_odometry = registry.odometry.len();

    println!();
    println!(
        "  {}. All datasets - {} files ({} g2o + {} BA problems)",
        ba_len + 5,
        total_odometry + total_ba,
        total_odometry,
        total_ba
    );
}

// ---------------------------------------------------------------------------
// Download functions
// ---------------------------------------------------------------------------

fn download_odometry_by_category(
    registry: &DatasetRegistry,
    category: &str,
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let mut success_count = 0;
    let mut fail_count = 0;
    let mut total_bytes = 0u64;

    for (name, entry) in registry.odometry_by_category(category) {
        let category_dir = base_output.join(&entry.category);
        std::fs::create_dir_all(&category_dir)?;
        let output_path = category_dir.join(&entry.filename);
        print!("  {name} ... ");
        std::io::stdout().flush()?;

        match download_and_size(&entry.url, &output_path) {
            Ok(size) => {
                total_bytes += size;
                success_count += 1;
                println!("OK ({size} bytes)");
            }
            Err(e) => {
                fail_count += 1;
                println!("FAILED: {e}");
            }
        }
    }
    Ok((success_count, fail_count, total_bytes))
}

fn download_all_g2o(
    registry: &DatasetRegistry,
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let (s1, f1, b1) = download_odometry_by_category(registry, "3d", base_output)?;
    let (s2, f2, b2) = download_odometry_by_category(registry, "2d", base_output)?;
    Ok((s1 + s2, f1 + f2, b1 + b2))
}

fn download_single_ba_dataset(
    name: &str,
    registry: &DatasetRegistry,
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let entry = registry
        .bundle_adjustment
        .get(name)
        .ok_or_else(|| format!("BA dataset '{name}' not found in registry"))?;

    let mut success_count = 0;
    let mut fail_count = 0;
    let mut total_bytes = 0u64;

    let dataset_path = base_output.join(name);

    for [cameras, points] in &entry.problems {
        let compressed_path =
            dataset_path.join(format!("problem-{cameras}-{points}-pre.txt.bz2"));
        let decompressed_path = dataset_path.join(format!("problem-{cameras}-{points}-pre.txt"));
        let url = entry.problem_url(*cameras, *points);

        print!("  problem-{cameras}-{points} ... ");
        std::io::stdout().flush()?;

        match download_and_size(&url, &compressed_path) {
            Ok(size) => {
                total_bytes += size;
                match decompress_and_cleanup(&compressed_path, &decompressed_path) {
                    Ok(()) => {
                        success_count += 1;
                        println!("OK ({size} bytes)");
                    }
                    Err(e) => {
                        fail_count += 1;
                        println!("DECOMPRESS FAILED: {e}");
                    }
                }
            }
            Err(e) => {
                fail_count += 1;
                println!("FAILED: {e}");
            }
        }
    }

    Ok((success_count, fail_count, total_bytes))
}

fn download_largest_each_ba(
    registry: &DatasetRegistry,
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let mut success_count = 0;
    let mut fail_count = 0;
    let mut total_bytes = 0u64;

    for (name, entry) in registry.ba_sorted() {
        let Some([cameras, points]) = entry.largest() else {
            continue;
        };

        let dataset_path = base_output.join(name);
        let compressed_path =
            dataset_path.join(format!("problem-{cameras}-{points}-pre.txt.bz2"));
        let decompressed_path = dataset_path.join(format!("problem-{cameras}-{points}-pre.txt"));
        let url = entry.problem_url(cameras, points);

        print!("  {name} largest (problem-{cameras}-{points}) ... ");
        std::io::stdout().flush()?;

        match download_and_size(&url, &compressed_path) {
            Ok(size) => {
                total_bytes += size;
                match decompress_and_cleanup(&compressed_path, &decompressed_path) {
                    Ok(()) => {
                        success_count += 1;
                        println!("OK ({size} bytes)");
                    }
                    Err(e) => {
                        fail_count += 1;
                        println!("DECOMPRESS FAILED: {e}");
                    }
                }
            }
            Err(e) => {
                fail_count += 1;
                println!("FAILED: {e}");
            }
        }
    }

    Ok((success_count, fail_count, total_bytes))
}

fn download_all_ba(
    registry: &DatasetRegistry,
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let mut total_success = 0;
    let mut total_fail = 0;
    let mut total_bytes = 0u64;

    for (name, _) in registry.ba_sorted() {
        let (s, f, b) = download_single_ba_dataset(name, registry, base_output)?;
        total_success += s;
        total_fail += f;
        total_bytes += b;
    }

    Ok((total_success, total_fail, total_bytes))
}

fn download_all_datasets(
    registry: &DatasetRegistry,
    base_output_odometry: &PathBuf,
    base_output_ba: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let (s1, f1, b1) = download_all_g2o(registry, base_output_odometry)?;
    let (s2, f2, b2) = download_all_ba(registry, base_output_ba)?;
    Ok((s1 + s2, f1 + f2, b1 + b2))
}

// ---------------------------------------------------------------------------
// Interactive prompt
// ---------------------------------------------------------------------------

fn get_user_selection(registry: &DatasetRegistry) -> Result<usize, Box<dyn std::error::Error>> {
    list_datasets(registry);
    println!();

    let ba_count = registry.bundle_adjustment.len();
    let max = ba_count + 5;
    print!("Enter your selection (0-{max}): ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    let selection: usize = input
        .trim()
        .parse()
        .map_err(|_| format!("Invalid input. Please enter a number between 0 and {max}."))?;

    if selection > max {
        return Err(format!("Invalid selection. Please enter a number between 0 and {max}.").into());
    }

    Ok(selection)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let registry = DatasetRegistry::load();

    if args.list {
        list_datasets(&registry);
        return Ok(());
    }

    let ba_sorted = registry.ba_sorted();
    let ba_count = ba_sorted.len(); // currently 5 (ladybug, trafalgar, dubrovnik, venice, final)

    let selection = match args.select {
        Some(s) => {
            let max = ba_count + 5;
            if s > max {
                println!("Invalid selection: {s}. Please enter a number between 0 and {max}.");
                return Ok(());
            }
            s
        }
        None => get_user_selection(&registry)?,
    };

    println!("\n=== Dataset Downloader ===");
    println!("Selected: {selection}");

    let odometry_output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(ODOMETRY_DATA_DIR));
    let ba_output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(BUNDLE_ADJUSTMENT_DATA_DIR));

    let mut total_success = 0;
    let mut total_fail = 0;
    let mut total_bytes = 0u64;

    match selection {
        1 => {
            println!("\n--- Downloading 3D g2o datasets ---");
            println!("Output: {:?}\n", odometry_output.join("3d"));
            let (s, f, b) = download_odometry_by_category(&registry, "3d", &odometry_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        2 => {
            println!("\n--- Downloading 2D g2o datasets ---");
            println!("Output: {:?}\n", odometry_output.join("2d"));
            let (s, f, b) = download_odometry_by_category(&registry, "2d", &odometry_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        3 => {
            println!("\n--- Downloading all odometry g2o datasets ---");
            println!("Output: {:?} and {:?}\n", odometry_output.join("2d"), odometry_output.join("3d"));
            let (s, f, b) = download_all_g2o(&registry, &odometry_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        n if n >= 4 && n <= ba_count + 3 => {
            let ds_idx = n - 4;
            let (name, _) = ba_sorted[ds_idx];
            println!("\n--- Downloading {name} ---");
            println!("Output: {ba_output:?}\n");
            let (s, f, b) = download_single_ba_dataset(name, &registry, &ba_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        n if n == ba_count + 4 => {
            println!("\n--- Downloading largest problem from each BA dataset ---");
            println!("Output: {ba_output:?}\n");
            let (s, f, b) = download_largest_each_ba(&registry, &ba_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        n if n == ba_count + 5 => {
            println!("\n--- Downloading ALL datasets ---");
            println!("Odometry output: {odometry_output:?}");
            println!("BA output:       {ba_output:?}\n");
            let (s, f, b) = download_all_datasets(&registry, &odometry_output, &ba_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        _ => {
            println!("Invalid selection: {selection}. Run with --list to see available options.");
            return Ok(());
        }
    }

    println!("\n=== Download Complete ===");
    println!("Success: {total_success}");
    println!("Failed:  {total_fail}");
    println!(
        "Total:   {total_bytes} bytes ({:.2} MB)",
        total_bytes as f64 / 1_048_576.0
    );

    if total_fail > 0 {
        std::process::exit(1);
    }

    Ok(())
}
