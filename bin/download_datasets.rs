//! Dataset downloader for pose graph and bundle adjustment datasets.
//!
//! Downloads datasets from two sources:
//! 1. **Luca Carlone's pose graph datasets** (3D/2D g2o): https://lucacarlone.mit.edu/datasets/
//! 2. **UW BAL bundle adjustment datasets**: https://grail.cs.washington.edu/projects/bal/
//!
//! Usage:
//! ```bash
//! # Download specific dataset by number
//! cargo run --bin download_datasets -- --select 3    # All odometry g2o
//! cargo run --bin download_datasets -- --select 0    # Largest from each BA
//!
//! # List available datasets
//! cargo run --bin download_datasets -- --list
//!
//! # Interactive mode (prompts for selection)
//! cargo run --bin download_datasets
//! ```

use clap::Parser;
use std::io::{Read, Write};
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

    /// Output directory (default: data/odometry or data/bundle_adjustment)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Skip confirmation prompt
    #[arg(short = 'y', long, default_value = "false")]
    yes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum OdometryCategory {
    D3G2o,
    D2G2o,
}

#[derive(Debug)]
struct OdometryDataset {
    name: &'static str,
    url: &'static str,
    filename: &'static str,
    category: OdometryCategory,
}

const ODOMETRY_DATASETS: &[OdometryDataset] = &[
    // 3D g2o format datasets (Carlone)
    OdometryDataset {
        name: "sphere_bignoise",
        url: "https://www.dropbox.com/s/ej5hb1ckcp3x42u/sphere_bignoise_vertex3.g2o?dl=1",
        filename: "sphere_bignoise_vertex3.g2o",
        category: OdometryCategory::D3G2o,
    },
    OdometryDataset {
        name: "torus3D",
        url: "https://www.dropbox.com/s/o95o2lbvww1100r/torus3D.g2o?dl=1",
        filename: "torus3D.g2o",
        category: OdometryCategory::D3G2o,
    },
    OdometryDataset {
        name: "grid3D",
        url: "https://www.dropbox.com/s/gjw9xl3t632gbrk/grid3D.g2o?dl=1",
        filename: "grid3D.g2o",
        category: OdometryCategory::D3G2o,
    },
    OdometryDataset {
        name: "parking-garage",
        url: "https://www.dropbox.com/s/zu23p8d522qccor/parking-garage.g2o?dl=1",
        filename: "parking-garage.g2o",
        category: OdometryCategory::D3G2o,
    },
    OdometryDataset {
        name: "cubicle",
        url: "https://www.dropbox.com/s/twpqdfphdw4md94/cubicle.g2o?dl=1",
        filename: "cubicle.g2o",
        category: OdometryCategory::D3G2o,
    },
    OdometryDataset {
        name: "rim",
        url: "https://www.dropbox.com/s/25qijwvfpmzh257/rim.g2o?dl=1",
        filename: "rim.g2o",
        category: OdometryCategory::D3G2o,
    },
    // 2D g2o format datasets (Carlone)
    OdometryDataset {
        name: "INTEL_g2o",
        url: "https://www.dropbox.com/s/vcz8cag7bo0zlaj/input_INTEL_g2o.g2o?dl=1",
        filename: "input_INTEL_g2o.g2o",
        category: OdometryCategory::D2G2o,
    },
    OdometryDataset {
        name: "MITb_g2o",
        url: "https://www.dropbox.com/s/d8fcn1jg1mebx8f/input_MITb_g2o.g2o?dl=1",
        filename: "input_MITb_g2o.g2o",
        category: OdometryCategory::D2G2o,
    },
    OdometryDataset {
        name: "M3500_g2o",
        url: "https://www.dropbox.com/s/gmdzo74b3tzvbrw/input_M3500_g2o.g2o?dl=1",
        filename: "input_M3500_g2o.g2o",
        category: OdometryCategory::D2G2o,
    },
    OdometryDataset {
        name: "M3500a_g2o",
        url: "https://www.dropbox.com/s/m9e866tdr2jlhf6/input_M3500a_g2o.g2o?dl=1",
        filename: "input_M3500a_g2o.g2o",
        category: OdometryCategory::D2G2o,
    },
    OdometryDataset {
        name: "M3500b_g2o",
        url: "https://www.dropbox.com/s/4tugr2hf8janxr3/input_M3500b_g2o.g2o?dl=1",
        filename: "input_M3500b_g2o.g2o",
        category: OdometryCategory::D2G2o,
    },
    OdometryDataset {
        name: "M3500c_g2o",
        url: "https://www.dropbox.com/s/6plsfzadfc1959r/input_M3500c_g2o.g2o?dl=1",
        filename: "input_M3500c_g2o.g2o",
        category: OdometryCategory::D2G2o,
    },
];

#[derive(Debug)]
struct BundleAdjustmentDataset {
    name: &'static str,
    url_prefix: &'static str,
    problems: &'static [(u32, u32)],
}

const BUNDLE_ADJUSTMENT_DATASETS: &[BundleAdjustmentDataset] = &[
    BundleAdjustmentDataset {
        name: "ladybug",
        url_prefix: "https://grail.cs.washington.edu/projects/bal/data/ladybug/problem",
        problems: &[
            (49, 7776),
            (73, 11032),
            (138, 19878),
            (318, 41628),
            (372, 47423),
            (412, 52215),
            (460, 56811),
            (539, 65220),
            (598, 69218),
            (646, 73584),
            (707, 78455),
            (783, 84444),
            (810, 88814),
            (856, 93344),
            (885, 97473),
            (931, 102699),
            (969, 105826),
            (1031, 110968),
            (1064, 113655),
            (1118, 118384),
            (1152, 122269),
            (1197, 126327),
            (1235, 129634),
            (1266, 132593),
            (1340, 137079),
            (1469, 145199),
            (1514, 147317),
            (1587, 150845),
            (1642, 153820),
            (1695, 155710),
            (1723, 156502),
        ],
    },
    BundleAdjustmentDataset {
        name: "trafalgar",
        url_prefix: "https://grail.cs.washington.edu/projects/bal/data/trafalgar/problem",
        problems: &[
            (21, 11315),
            (39, 18060),
            (50, 20431),
            (126, 40037),
            (138, 44033),
            (161, 48126),
            (170, 49267),
            (174, 50489),
            (193, 53101),
            (201, 54427),
            (206, 54562),
            (215, 55910),
            (225, 57665),
            (257, 65132),
        ],
    },
    BundleAdjustmentDataset {
        name: "dubrovnik",
        url_prefix: "https://grail.cs.washington.edu/projects/bal/data/dubrovnik/problem",
        problems: &[
            (16, 22106),
            (88, 64298),
            (135, 90642),
            (142, 93602),
            (150, 95821),
            (161, 103832),
            (173, 111908),
            (182, 116770),
            (202, 132796),
            (237, 154414),
            (253, 163691),
            (262, 169354),
            (273, 176305),
            (287, 182023),
            (308, 195089),
            (356, 226730),
        ],
    },
    BundleAdjustmentDataset {
        name: "venice",
        url_prefix: "https://grail.cs.washington.edu/projects/bal/data/venice/problem",
        problems: &[
            (52, 64053),
            (89, 110973),
            (245, 198739),
            (427, 310384),
            (744, 543562),
            (951, 708276),
            (1102, 780462),
            (1158, 802917),
            (1184, 816583),
            (1238, 843534),
            (1288, 866452),
            (1350, 894716),
            (1408, 912229),
            (1425, 916895),
            (1473, 930345),
            (1490, 935273),
            (1521, 939551),
            (1544, 942409),
            (1638, 976803),
            (1666, 983911),
            (1672, 986962),
            (1681, 983415),
            (1682, 983268),
            (1684, 983269),
            (1695, 984689),
            (1696, 984816),
            (1706, 985529),
            (1776, 993909),
            (1778, 993923),
        ],
    },
    BundleAdjustmentDataset {
        name: "final",
        url_prefix: "https://grail.cs.washington.edu/projects/bal/data/final/problem",
        problems: &[
            (93, 61203),
            (394, 100368),
            (871, 527480),
            (961, 187103),
            (1936, 649673),
            (3068, 310854),
            (4585, 1324582),
            (13682, 4456117),
        ],
    },
];

fn odometry_category_dir(category: OdometryCategory) -> &'static str {
    match category {
        OdometryCategory::D3G2o => "3D_g2o",
        OdometryCategory::D2G2o => "2D_g2o",
    }
}

fn download_file(url: &str, output_path: &PathBuf) -> Result<u64, Box<dyn std::error::Error>> {
    println!("  Downloading: {}", url);
    println!("  Saving to: {:?}", output_path);

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Failed to read response: {}", e))?;

    let size = bytes.len() as u64;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(output_path, &bytes)?;

    Ok(size)
}

fn decompress_bzip2(
    input_path: &PathBuf,
    output_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use bzip2::read::BzDecoder;

    println!("  Decompressing: {:?}", input_path);
    println!("  Extracting to: {:?}", output_path);

    let compressed_data = std::fs::read(input_path)?;
    let mut decoder = BzDecoder::new(compressed_data.as_slice());
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(output_path, &decompressed)?;
    println!("  Done!");
    Ok(())
}

fn list_datasets() {
    println!("\n=== Available Datasets ===\n");

    println!("Odometry (g2o format):");
    println!(
        "  1. 3D g2o      - 6 files (sphere_bignoise, torus3D, grid3D, parking-garage, cubicle, rim)"
    );
    println!("  2. 2D g2o      - 6 files (intel, MITb, M3500, M3500a, M3500b, M3500c)");
    println!("  3. All odometry g2o - 12 files (all odometry datasets)");
    println!();

    println!("Bundle Adjustment (UW BAL format):");
    let mut idx = 4;
    for dataset in BUNDLE_ADJUSTMENT_DATASETS {
        let largest = dataset.problems.last().unwrap();
        println!(
            "  {}. {:12} - {} problems (largest: {}x{})",
            idx,
            dataset.name,
            dataset.problems.len(),
            largest.0,
            largest.1
        );
        idx += 1;
    }
    println!("  9. Largest each  - 5 problems (largest from each BA)");

    let total_ba = BUNDLE_ADJUSTMENT_DATASETS
        .iter()
        .map(|d| d.problems.len())
        .sum::<usize>();
    let total_odometry = ODOMETRY_DATASETS.len();

    println!();
    println!(
        "  10. All datasets - {} files ({} g2o + {} BA problems)",
        total_odometry + total_ba,
        total_odometry,
        total_ba
    );
}

fn download_3d_g2o(
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let mut success_count = 0;
    let mut fail_count = 0;
    let mut total_bytes = 0u64;

    let category_path = base_output.join(odometry_category_dir(OdometryCategory::D3G2o));
    for ds in ODOMETRY_DATASETS
        .iter()
        .filter(|d| d.category == OdometryCategory::D3G2o)
    {
        let output_path = category_path.join(ds.filename);
        print!("  {} ... ", ds.name);
        std::io::stdout().flush()?;

        match download_file(ds.url, &output_path) {
            Ok(size) => {
                total_bytes += size;
                success_count += 1;
                println!("OK ({} bytes)", size);
            }
            Err(e) => {
                fail_count += 1;
                println!("FAILED: {}", e);
            }
        }
    }
    Ok((success_count, fail_count, total_bytes))
}

fn download_2d_g2o(
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let mut success_count = 0;
    let mut fail_count = 0;
    let mut total_bytes = 0u64;

    let category_path = base_output.join(odometry_category_dir(OdometryCategory::D2G2o));
    for ds in ODOMETRY_DATASETS
        .iter()
        .filter(|d| d.category == OdometryCategory::D2G2o)
    {
        let output_path = category_path.join(ds.filename);
        print!("  {} ... ", ds.name);
        std::io::stdout().flush()?;

        match download_file(ds.url, &output_path) {
            Ok(size) => {
                total_bytes += size;
                success_count += 1;
                println!("OK ({} bytes)", size);
            }
            Err(e) => {
                fail_count += 1;
                println!("FAILED: {}", e);
            }
        }
    }
    Ok((success_count, fail_count, total_bytes))
}

fn download_all_g2o(
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let (s1, f1, b1) = download_3d_g2o(base_output)?;
    let (s2, f2, b2) = download_2d_g2o(base_output)?;
    Ok((s1 + s2, f1 + f2, b1 + b2))
}

fn download_single_ba_dataset(
    dataset: &BundleAdjustmentDataset,
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let mut success_count = 0;
    let mut fail_count = 0;
    let mut total_bytes = 0u64;

    let dataset_path = base_output.join(dataset.name);

    for (cameras, points) in dataset.problems {
        let filename = format!("problem-{}-{}-pre.txt.bz2", cameras, points);
        let url = format!("{}-{}-{}-pre.txt.bz2", dataset.url_prefix, cameras, points);

        let compressed_path = dataset_path.join(&filename);
        let decompressed_filename = format!("problem-{}-{}-pre.txt", cameras, points);
        let decompressed_path = dataset_path.join(&decompressed_filename);

        print!("  problem-{}-{} ... ", cameras, points);
        std::io::stdout().flush()?;

        match download_file(&url, &compressed_path) {
            Ok(size) => {
                total_bytes += size;
                match decompress_bzip2(&compressed_path, &decompressed_path) {
                    Ok(()) => {
                        success_count += 1;
                        println!("OK ({} bytes)", size);
                        if let Err(e) = std::fs::remove_file(&compressed_path) {
                            eprintln!("  Warning: could not remove compressed file: {}", e);
                        }
                    }
                    Err(e) => {
                        fail_count += 1;
                        println!("DECOMPRESS FAILED: {}", e);
                    }
                }
            }
            Err(e) => {
                fail_count += 1;
                println!("FAILED: {}", e);
            }
        }
    }

    Ok((success_count, fail_count, total_bytes))
}

fn download_largest_each_ba(
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let mut success_count = 0;
    let mut fail_count = 0;
    let mut total_bytes = 0u64;

    for dataset in BUNDLE_ADJUSTMENT_DATASETS {
        let largest = dataset.problems.last().unwrap();
        let filename = format!("problem-{}-{}-pre.txt.bz2", largest.0, largest.1);
        let url = format!(
            "{}-{}-{}-pre.txt.bz2",
            dataset.url_prefix, largest.0, largest.1
        );

        let dataset_path = base_output.join(dataset.name);
        let compressed_path = dataset_path.join(&filename);
        let decompressed_filename = format!("problem-{}-{}-pre.txt", largest.0, largest.1);
        let decompressed_path = dataset_path.join(&decompressed_filename);

        print!(
            "  {} largest (problem-{}-{}) ... ",
            dataset.name, largest.0, largest.1
        );
        std::io::stdout().flush()?;

        match download_file(&url, &compressed_path) {
            Ok(size) => {
                total_bytes += size;
                match decompress_bzip2(&compressed_path, &decompressed_path) {
                    Ok(()) => {
                        success_count += 1;
                        println!("OK ({} bytes)", size);
                        if let Err(e) = std::fs::remove_file(&compressed_path) {
                            eprintln!("  Warning: could not remove compressed file: {}", e);
                        }
                    }
                    Err(e) => {
                        fail_count += 1;
                        println!("DECOMPRESS FAILED: {}", e);
                    }
                }
            }
            Err(e) => {
                fail_count += 1;
                println!("FAILED: {}", e);
            }
        }
    }

    Ok((success_count, fail_count, total_bytes))
}

fn download_all_ba(
    base_output: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let mut total_success = 0;
    let mut total_fail = 0;
    let mut total_bytes = 0u64;

    for dataset in BUNDLE_ADJUSTMENT_DATASETS {
        let (s, f, b) = download_single_ba_dataset(dataset, base_output)?;
        total_success += s;
        total_fail += f;
        total_bytes += b;
    }

    Ok((total_success, total_fail, total_bytes))
}

fn download_all_datasets(
    base_output_odometry: &PathBuf,
    base_output_ba: &PathBuf,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let (s1, f1, b1) = download_all_g2o(base_output_odometry)?;
    let (s2, f2, b2) = download_all_ba(base_output_ba)?;
    Ok((s1 + s2, f1 + f2, b1 + b2))
}

fn get_user_selection() -> Result<usize, Box<dyn std::error::Error>> {
    list_datasets();
    println!();
    print!("Enter your selection (0-10): ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    let selection: usize = input
        .trim()
        .parse()
        .map_err(|_| "Invalid input. Please enter a number between 0 and 10.")?;

    if selection > 10 {
        return Err("Invalid selection. Please enter a number between 0 and 10.".into());
    }

    Ok(selection)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.list {
        list_datasets();
        return Ok(());
    }

    let selection = match args.select {
        Some(s) => {
            if s > 10 {
                println!(
                    "Invalid selection: {}. Please enter a number between 0 and 10.",
                    s
                );
                return Ok(());
            }
            s
        }
        None => get_user_selection()?,
    };

    println!("\n=== Dataset Downloader ===");
    println!("Selected: {}", selection);

    let odometry_output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("data/odometry"));
    let ba_output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("data/bundle_adjustment"));

    let mut total_success = 0;
    let mut total_fail = 0;
    let mut total_bytes = 0u64;

    match selection {
        1 => {
            println!("\n--- Downloading 3D g2o datasets ---");
            println!("Output: {:?}\n", odometry_output);
            let (s, f, b) = download_3d_g2o(&odometry_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        2 => {
            println!("\n--- Downloading 2D g2o datasets ---");
            println!("Output: {:?}\n", odometry_output);
            let (s, f, b) = download_2d_g2o(&odometry_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        3 => {
            println!("\n--- Downloading all odometry g2o datasets ---");
            println!("Output: {:?}\n", odometry_output);
            let (s, f, b) = download_all_g2o(&odometry_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        4 => {
            println!("\n--- Downloading Ladybug ---");
            println!("Output: {:?}\n", ba_output);
            let (s, f, b) = download_single_ba_dataset(&BUNDLE_ADJUSTMENT_DATASETS[0], &ba_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        5 => {
            println!("\n--- Downloading Trafalgar ---");
            println!("Output: {:?}\n", ba_output);
            let (s, f, b) = download_single_ba_dataset(&BUNDLE_ADJUSTMENT_DATASETS[1], &ba_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        6 => {
            println!("\n--- Downloading Dubrovnik ---");
            println!("Output: {:?}\n", ba_output);
            let (s, f, b) = download_single_ba_dataset(&BUNDLE_ADJUSTMENT_DATASETS[2], &ba_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        7 => {
            println!("\n--- Downloading Venice ---");
            println!("Output: {:?}\n", ba_output);
            let (s, f, b) = download_single_ba_dataset(&BUNDLE_ADJUSTMENT_DATASETS[3], &ba_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        8 => {
            println!("\n--- Downloading Final ---");
            println!("Output: {:?}\n", ba_output);
            let (s, f, b) = download_single_ba_dataset(&BUNDLE_ADJUSTMENT_DATASETS[4], &ba_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        9 => {
            println!("\n--- Downloading largest problem from each BA dataset ---");
            println!("Output: {:?}\n", ba_output);
            let (s, f, b) = download_largest_each_ba(&ba_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        10 => {
            println!("\n--- Downloading ALL datasets ---");
            println!("Odometry output: {:?}", odometry_output);
            println!("BA output:       {:?}\n", ba_output);
            let (s, f, b) = download_all_datasets(&odometry_output, &ba_output)?;
            total_success += s;
            total_fail += f;
            total_bytes += b;
        }
        _ => {
            println!(
                "Invalid selection: {}. Run with --list to see available options.",
                selection
            );
            return Ok(());
        }
    }

    println!("\n=== Download Complete ===");
    println!("Success: {}", total_success);
    println!("Failed:  {}", total_fail);
    println!(
        "Total:   {} bytes ({} MB)",
        total_bytes,
        total_bytes as f64 / 1_048_576.0
    );

    if total_fail > 0 {
        std::process::exit(1);
    }

    Ok(())
}
