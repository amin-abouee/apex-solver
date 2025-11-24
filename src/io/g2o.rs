use crate::io::{EdgeSE2, EdgeSE3, Graph, GraphLoader, IoError, VertexSE2, VertexSE3};
use memmap2;
use rayon::prelude::*;
use std::collections::HashMap;
use std::{fs::File, io::Write, path::Path};

/// High-performance G2O file loader
pub struct G2oLoader;

impl GraphLoader for G2oLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<Graph, IoError> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|e| {
            IoError::Io(e).log_with_source(format!("Failed to open G2O file: {:?}", path_ref))
        })?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| {
                IoError::Io(e)
                    .log_with_source(format!("Failed to memory-map G2O file: {:?}", path_ref))
            })?
        };
        let content = std::str::from_utf8(&mmap).map_err(|e| {
            IoError::Parse {
                line: 0,
                message: format!("Invalid UTF-8: {e}"),
            }
            .log()
        })?;

        Self::parse_content(content)
    }

    fn write<P: AsRef<Path>>(graph: &Graph, path: P) -> Result<(), IoError> {
        let path_ref = path.as_ref();
        let mut file = File::create(path_ref).map_err(|e| {
            IoError::Io(e).log_with_source(format!("Failed to create G2O file: {:?}", path_ref))
        })?;

        // Write header comment
        writeln!(file, "# G2O file written by Apex Solver")
            .map_err(|e| IoError::Io(e).log_with_source("Failed to write G2O header"))?;
        writeln!(
            file,
            "# Timestamp: {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        )
        .map_err(|e| IoError::Io(e).log_with_source("Failed to write G2O timestamp"))?;
        writeln!(
            file,
            "# SE2 vertices: {}, SE3 vertices: {}, SE2 edges: {}, SE3 edges: {}",
            graph.vertices_se2.len(),
            graph.vertices_se3.len(),
            graph.edges_se2.len(),
            graph.edges_se3.len()
        )
        .map_err(|e| IoError::Io(e).log_with_source("Failed to write G2O statistics"))?;
        writeln!(file)
            .map_err(|e| IoError::Io(e).log_with_source("Failed to write G2O header newline"))?;

        // Write SE2 vertices (sorted by ID)
        let mut se2_ids: Vec<_> = graph.vertices_se2.keys().collect();
        se2_ids.sort();

        for id in se2_ids {
            let vertex = &graph.vertices_se2[id];
            writeln!(
                file,
                "VERTEX_SE2 {} {:.17e} {:.17e} {:.17e}",
                vertex.id,
                vertex.x(),
                vertex.y(),
                vertex.theta()
            )
            .map_err(|e| {
                IoError::Io(e).log_with_source(format!("Failed to write SE2 vertex {}", vertex.id))
            })?;
        }

        // Write SE3 vertices (sorted by ID)
        let mut se3_ids: Vec<_> = graph.vertices_se3.keys().collect();
        se3_ids.sort();

        for id in se3_ids {
            let vertex = &graph.vertices_se3[id];
            let trans = vertex.translation();
            let quat = vertex.rotation();
            writeln!(
                file,
                "VERTEX_SE3:QUAT {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                vertex.id, trans.x, trans.y, trans.z, quat.i, quat.j, quat.k, quat.w
            )
            .map_err(|e| {
                IoError::Io(e).log_with_source(format!("Failed to write SE3 vertex {}", vertex.id))
            })?;
        }

        // Write SE2 edges
        for edge in &graph.edges_se2 {
            let meas = &edge.measurement;
            let info = &edge.information;

            // G2O SE2 information matrix order: i11, i12, i22, i33, i13, i23
            writeln!(
                file,
                "EDGE_SE2 {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                edge.from,
                edge.to,
                meas.x(),
                meas.y(),
                meas.angle(),
                info[(0, 0)],
                info[(0, 1)],
                info[(1, 1)],
                info[(2, 2)],
                info[(0, 2)],
                info[(1, 2)]
            )
            .map_err(|e| {
                IoError::Io(e).log_with_source(format!(
                    "Failed to write SE2 edge {} -> {}",
                    edge.from, edge.to
                ))
            })?;
        }

        // Write SE3 edges
        for edge in &graph.edges_se3 {
            let trans = edge.measurement.translation();
            let quat = edge.measurement.rotation_quaternion();
            let info = &edge.information;

            // Write EDGE_SE3:QUAT with full 6x6 upper triangular information matrix (21 values)
            write!(
                file,
                "EDGE_SE3:QUAT {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                edge.from, edge.to, trans.x, trans.y, trans.z, quat.i, quat.j, quat.k, quat.w
            )
            .map_err(|e| {
                IoError::Io(e).log_with_source(format!(
                    "Failed to write SE3 edge {} -> {}",
                    edge.from, edge.to
                ))
            })?;

            // Write upper triangular information matrix (21 values)
            for i in 0..6 {
                for j in i..6 {
                    write!(file, " {:.17e}", info[(i, j)]).map_err(|e| {
                        IoError::Io(e).log_with_source(format!(
                            "Failed to write SE3 edge {} -> {} information matrix",
                            edge.from, edge.to
                        ))
                    })?;
                }
            }
            writeln!(file).map_err(|e| {
                IoError::Io(e).log_with_source(format!(
                    "Failed to write SE3 edge {} -> {} newline",
                    edge.from, edge.to
                ))
            })?;
        }

        Ok(())
    }
}

impl G2oLoader {
    /// Parse G2O content with performance optimizations
    fn parse_content(content: &str) -> Result<Graph, IoError> {
        let lines: Vec<&str> = content.lines().collect();
        let minimum_lines_for_parallel = 1000;

        // Pre-allocate collections based on estimated size
        let estimated_vertices = lines.len() / 4;
        let estimated_edges = estimated_vertices * 3;
        let mut graph = Graph {
            vertices_se2: HashMap::with_capacity(estimated_vertices),
            vertices_se3: HashMap::with_capacity(estimated_vertices),
            edges_se2: Vec::with_capacity(estimated_edges),
            edges_se3: Vec::with_capacity(estimated_edges),
        };

        // For large files, use parallel processing
        if lines.len() > minimum_lines_for_parallel {
            Self::parse_parallel(&lines, &mut graph)?;
        } else {
            Self::parse_sequential(&lines, &mut graph)?;
        }

        Ok(graph)
    }

    /// Sequential parsing for smaller files
    fn parse_sequential(lines: &[&str], graph: &mut Graph) -> Result<(), IoError> {
        for (line_num, line) in lines.iter().enumerate() {
            Self::parse_line(line, line_num + 1, graph)?;
        }
        Ok(())
    }

    /// Parallel parsing for larger files
    fn parse_parallel(lines: &[&str], graph: &mut Graph) -> Result<(), IoError> {
        // Collect parse results in parallel
        let results: Result<Vec<_>, IoError> = lines
            .par_iter()
            .enumerate()
            .map(|(line_num, line)| Self::parse_line_to_enum(line, line_num + 1))
            .collect();

        let parsed_items = results?;

        // Sequential insertion to avoid concurrent modification
        for item in parsed_items.into_iter().flatten() {
            match item {
                ParsedItem::VertexSE2(vertex) => {
                    let id = vertex.id;
                    if graph.vertices_se2.insert(id, vertex).is_some() {
                        return Err(IoError::DuplicateVertex { id });
                    }
                }
                ParsedItem::VertexSE3(vertex) => {
                    let id = vertex.id;
                    if graph.vertices_se3.insert(id, vertex).is_some() {
                        return Err(IoError::DuplicateVertex { id });
                    }
                }
                ParsedItem::EdgeSE2(edge) => {
                    graph.edges_se2.push(edge);
                }
                ParsedItem::EdgeSE3(edge) => {
                    graph.edges_se3.push(*edge);
                }
            }
        }

        Ok(())
    }

    /// Parse a single line (for sequential processing)
    fn parse_line(line: &str, line_num: usize, graph: &mut Graph) -> Result<(), IoError> {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            return Ok(());
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        match parts[0] {
            "VERTEX_SE2" => {
                let vertex = Self::parse_vertex_se2(&parts, line_num)?;
                let id = vertex.id;
                if graph.vertices_se2.insert(id, vertex).is_some() {
                    return Err(IoError::DuplicateVertex { id });
                }
            }
            "VERTEX_SE3:QUAT" => {
                let vertex = Self::parse_vertex_se3(&parts, line_num)?;
                let id = vertex.id;
                if graph.vertices_se3.insert(id, vertex).is_some() {
                    return Err(IoError::DuplicateVertex { id });
                }
            }
            "EDGE_SE2" => {
                let edge = Self::parse_edge_se2(&parts, line_num)?;
                graph.edges_se2.push(edge);
            }
            "EDGE_SE3:QUAT" => {
                let edge = Self::parse_edge_se3(&parts, line_num)?;
                graph.edges_se3.push(edge);
            }
            _ => {
                // Skip unknown types silently for compatibility
            }
        }

        Ok(())
    }

    /// Parse a single line to enum (for parallel processing)
    fn parse_line_to_enum(line: &str, line_num: usize) -> Result<Option<ParsedItem>, IoError> {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(None);
        }

        let item = match parts[0] {
            "VERTEX_SE2" => Some(ParsedItem::VertexSE2(Self::parse_vertex_se2(
                &parts, line_num,
            )?)),
            "VERTEX_SE3:QUAT" => Some(ParsedItem::VertexSE3(Self::parse_vertex_se3(
                &parts, line_num,
            )?)),
            "EDGE_SE2" => Some(ParsedItem::EdgeSE2(Self::parse_edge_se2(&parts, line_num)?)),
            "EDGE_SE3:QUAT" => Some(ParsedItem::EdgeSE3(Box::new(Self::parse_edge_se3(
                &parts, line_num,
            )?))),
            _ => None, // Skip unknown types
        };

        Ok(item)
    }

    /// Parse VERTEX_SE2 line
    pub fn parse_vertex_se2(parts: &[&str], line_num: usize) -> Result<VertexSE2, IoError> {
        if parts.len() < 5 {
            return Err(IoError::MissingFields { line: line_num });
        }

        let id = parts[1]
            .parse::<usize>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[1].to_string(),
            })?;

        let x = parts[2]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[2].to_string(),
            })?;

        let y = parts[3]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[3].to_string(),
            })?;

        let theta = parts[4]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[4].to_string(),
            })?;

        Ok(VertexSE2::new(id, x, y, theta))
    }

    /// Parse VERTEX_SE3:QUAT line
    pub fn parse_vertex_se3(parts: &[&str], line_num: usize) -> Result<VertexSE3, IoError> {
        if parts.len() < 9 {
            return Err(IoError::MissingFields { line: line_num });
        }

        let id = parts[1]
            .parse::<usize>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[1].to_string(),
            })?;

        let x = parts[2]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[2].to_string(),
            })?;

        let y = parts[3]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[3].to_string(),
            })?;

        let z = parts[4]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[4].to_string(),
            })?;

        let qx = parts[5]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[5].to_string(),
            })?;

        let qy = parts[6]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[6].to_string(),
            })?;

        let qz = parts[7]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[7].to_string(),
            })?;

        let qw = parts[8]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[8].to_string(),
            })?;

        let translation = nalgebra::Vector3::new(x, y, z);
        let quaternion = nalgebra::Quaternion::new(qw, qx, qy, qz);

        // Validate quaternion normalization
        let quat_norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
        if (quat_norm - 1.0).abs() > 0.01 {
            return Err(IoError::InvalidQuaternion {
                line: line_num,
                norm: quat_norm,
            });
        }

        // Always normalize for numerical safety
        let quaternion = quaternion.normalize();

        Ok(VertexSE3::from_translation_quaternion(
            id,
            translation,
            quaternion,
        ))
    }

    /// Parse EDGE_SE2 line
    fn parse_edge_se2(parts: &[&str], line_num: usize) -> Result<EdgeSE2, IoError> {
        if parts.len() < 12 {
            return Err(IoError::MissingFields { line: line_num });
        }

        let from = parts[1]
            .parse::<usize>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[1].to_string(),
            })?;

        let to = parts[2]
            .parse::<usize>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[2].to_string(),
            })?;

        // Parse measurement (dx, dy, dtheta)
        let dx = parts[3]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[3].to_string(),
            })?;
        let dy = parts[4]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[4].to_string(),
            })?;
        let dtheta = parts[5]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[5].to_string(),
            })?;

        // Parse information matrix (upper triangular: i11, i12, i13, i22, i23, i33)
        let info_values: Result<Vec<f64>, _> =
            parts[6..12].iter().map(|s| s.parse::<f64>()).collect();

        let info_values = info_values.map_err(|_| IoError::Parse {
            line: line_num,
            message: "Invalid information matrix values".to_string(),
        })?;

        let information = nalgebra::Matrix3::new(
            info_values[0],
            info_values[1],
            info_values[2],
            info_values[1],
            info_values[3],
            info_values[4],
            info_values[2],
            info_values[4],
            info_values[5],
        );

        Ok(EdgeSE2::new(from, to, dx, dy, dtheta, information))
    }

    /// Parse EDGE_SE3:QUAT line (placeholder implementation)
    fn parse_edge_se3(parts: &[&str], line_num: usize) -> Result<EdgeSE3, IoError> {
        // EDGE_SE3:QUAT from_id to_id tx ty tz qx qy qz qw [information matrix values]
        if parts.len() < 10 {
            return Err(IoError::MissingFields { line: line_num });
        }

        // Parse vertex IDs
        let from = parts[1]
            .parse::<usize>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[1].to_string(),
            })?;

        let to = parts[2]
            .parse::<usize>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[2].to_string(),
            })?;

        // Parse translation (tx, ty, tz)
        let tx = parts[3]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[3].to_string(),
            })?;

        let ty = parts[4]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[4].to_string(),
            })?;

        let tz = parts[5]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[5].to_string(),
            })?;

        let translation = nalgebra::Vector3::new(tx, ty, tz);

        // Parse rotation quaternion (qx, qy, qz, qw)
        let qx = parts[6]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[6].to_string(),
            })?;

        let qy = parts[7]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[7].to_string(),
            })?;

        let qz = parts[8]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[8].to_string(),
            })?;

        let qw = parts[9]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[9].to_string(),
            })?;

        let rotation =
            nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));

        // Parse information matrix (upper triangular: i11, i12, i13, i14, i15, i16, i22, i23, i24, i25, i26, i33, i34, i35, i36, i44, i45, i46, i55, i56, i66)
        let info_values: Result<Vec<f64>, _> =
            parts[10..31].iter().map(|s| s.parse::<f64>()).collect();

        let info_values = info_values.map_err(|_| IoError::Parse {
            line: line_num,
            message: "Invalid information matrix values".to_string(),
        })?;

        let information = nalgebra::Matrix6::new(
            info_values[0],
            info_values[1],
            info_values[2],
            info_values[3],
            info_values[4],
            info_values[5],
            info_values[1],
            info_values[6],
            info_values[7],
            info_values[8],
            info_values[9],
            info_values[10],
            info_values[2],
            info_values[7],
            info_values[11],
            info_values[12],
            info_values[13],
            info_values[14],
            info_values[3],
            info_values[8],
            info_values[12],
            info_values[15],
            info_values[16],
            info_values[17],
            info_values[4],
            info_values[9],
            info_values[13],
            info_values[16],
            info_values[18],
            info_values[19],
            info_values[5],
            info_values[10],
            info_values[14],
            info_values[17],
            info_values[19],
            info_values[20],
        );

        Ok(EdgeSE3::new(from, to, translation, rotation, information))
    }
}

/// Enum for parsed items (used in parallel processing)
enum ParsedItem {
    VertexSE2(VertexSE2),
    VertexSE3(VertexSE3),
    EdgeSE2(EdgeSE2),
    EdgeSE3(Box<EdgeSE3>),
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_parse_vertex_se2() -> TestResult {
        let parts = vec!["VERTEX_SE2", "0", "1.0", "2.0", "0.5"];
        let vertex = G2oLoader::parse_vertex_se2(&parts, 1)?;

        assert_eq!(vertex.id(), 0);
        assert_eq!(vertex.x(), 1.0);
        assert_eq!(vertex.y(), 2.0);
        assert_eq!(vertex.theta(), 0.5);

        Ok(())
    }

    #[test]
    fn test_parse_vertex_se3() -> TestResult {
        let parts = vec![
            "VERTEX_SE3:QUAT",
            "1",
            "1.0",
            "2.0",
            "3.0",
            "0.0",
            "0.0",
            "0.0",
            "1.0",
        ];
        let vertex = G2oLoader::parse_vertex_se3(&parts, 1)?;

        assert_eq!(vertex.id(), 1);
        assert_eq!(vertex.translation(), nalgebra::Vector3::new(1.0, 2.0, 3.0));
        assert!(vertex.rotation().quaternion().w > 0.99); // Should be identity quaternion

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        // Test invalid number
        let parts = vec!["VERTEX_SE2", "invalid", "1.0", "2.0", "0.5"];
        let result = G2oLoader::parse_vertex_se2(&parts, 1);
        assert!(matches!(result, Err(IoError::InvalidNumber { .. })));

        // Test missing fields
        let parts = vec!["VERTEX_SE2", "0"];
        let result = G2oLoader::parse_vertex_se2(&parts, 1);
        assert!(matches!(result, Err(IoError::MissingFields { .. })));
    }
}
