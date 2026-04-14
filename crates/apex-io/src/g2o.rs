use crate::{EdgeSE2, EdgeSE3, Graph, GraphLoader, IoError, VertexSE2, VertexSE3};
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
        // SAFETY: The file is opened read-only and the handle remains valid for the
        // lifetime of `mmap`. No other thread modifies the file during this scope.
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
    use nalgebra::{Matrix3, Matrix6, UnitQuaternion, Vector3};
    use std::io::Write;
    use tempfile::NamedTempFile;

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

    #[test]
    fn test_write_se2_graph_round_trip() -> TestResult {
        let mut graph = Graph::new();
        graph
            .vertices_se2
            .insert(0, VertexSE2::new(0, 1.0, 2.0, 0.5));
        graph
            .vertices_se2
            .insert(1, VertexSE2::new(1, 3.0, 4.0, 1.0));
        let info = Matrix3::new(500.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0, 200.0);
        graph
            .edges_se2
            .push(EdgeSE2::new(0, 1, 0.5, 0.3, 0.1, info));

        let f = NamedTempFile::new()?;
        G2oLoader::write(&graph, f.path())?;
        let loaded = G2oLoader::load(f.path())?;

        assert_eq!(loaded.vertices_se2.len(), 2);
        assert_eq!(loaded.edges_se2.len(), 1);
        let v0 = &loaded.vertices_se2[&0];
        assert!((v0.x() - 1.0).abs() < 1e-10);
        assert!((v0.y() - 2.0).abs() < 1e-10);
        let e = &loaded.edges_se2[0];
        assert_eq!(e.from, 0);
        assert_eq!(e.to, 1);
        assert!((e.information[(0, 0)] - 500.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_write_se3_graph_round_trip() -> TestResult {
        let trans = Vector3::new(1.0, 2.0, 3.0);
        let rot = UnitQuaternion::identity();
        let mut graph = Graph::new();
        graph.vertices_se3.insert(0, VertexSE3::new(0, trans, rot));
        graph
            .vertices_se3
            .insert(1, VertexSE3::new(1, Vector3::zeros(), rot));
        graph
            .edges_se3
            .push(EdgeSE3::new(0, 1, trans, rot, Matrix6::identity()));

        let f = NamedTempFile::new()?;
        G2oLoader::write(&graph, f.path())?;
        let loaded = G2oLoader::load(f.path())?;

        assert_eq!(loaded.vertices_se3.len(), 2);
        assert_eq!(loaded.edges_se3.len(), 1);
        let v0 = &loaded.vertices_se3[&0];
        assert!((v0.x() - 1.0).abs() < 1e-10);
        assert!((v0.y() - 2.0).abs() < 1e-10);
        assert!((v0.z() - 3.0).abs() < 1e-10);
        let e = &loaded.edges_se3[0];
        assert!((e.information[(0, 0)] - 1.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_write_mixed_graph_round_trip() -> TestResult {
        let mut graph = Graph::new();
        graph
            .vertices_se2
            .insert(0, VertexSE2::new(0, 1.0, 0.0, 0.0));
        graph.vertices_se3.insert(
            1,
            VertexSE3::new(1, Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity()),
        );

        let f = NamedTempFile::new()?;
        G2oLoader::write(&graph, f.path())?;
        let loaded = G2oLoader::load(f.path())?;

        assert_eq!(loaded.vertices_se2.len(), 1);
        assert_eq!(loaded.vertices_se3.len(), 1);
        Ok(())
    }

    #[test]
    fn test_write_empty_graph() -> TestResult {
        let graph = Graph::new();
        let f = NamedTempFile::new()?;
        G2oLoader::write(&graph, f.path())?;
        let loaded = G2oLoader::load(f.path())?;
        assert_eq!(loaded.vertices_se2.len(), 0);
        assert_eq!(loaded.vertices_se3.len(), 0);
        assert_eq!(loaded.edges_se2.len(), 0);
        assert_eq!(loaded.edges_se3.len(), 0);
        Ok(())
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = G2oLoader::load("/nonexistent/path/file.g2o");
        assert!(result.is_err(), "loading a missing file should return Err");
    }

    #[test]
    fn test_parse_vertex_se3_invalid_quaternion_norm() -> TestResult {
        // qw=0.1 gives norm ≈ 0.1, which is far from 1.0 (threshold: |norm-1| > 0.01)
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 0 0.0 0.0 0.0 0.0 0.0 0.0 0.1")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidQuaternion { .. })),
            "far-from-unit quaternion should return InvalidQuaternion"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se2_information_matrix() -> TestResult {
        // parse_edge_se2 reads info values as: i11, i12, i13, i22, i23, i33
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE2 0 0.0 0.0 0.0")?;
        writeln!(f, "VERTEX_SE2 1 1.0 0.0 0.0")?;
        writeln!(f, "EDGE_SE2 0 1 1.0 0.0 0.0 500.0 0.0 0.0 300.0 0.0 200.0")?;
        f.flush()?;
        let graph = G2oLoader::load(f.path())?;
        assert_eq!(graph.edges_se2.len(), 1);
        let e = &graph.edges_se2[0];
        assert_eq!(e.from, 0);
        assert_eq!(e.to, 1);
        assert!(
            (e.information[(0, 0)] - 500.0).abs() < 1e-10,
            "i11={}",
            e.information[(0, 0)]
        );
        assert!(
            (e.information[(1, 1)] - 300.0).abs() < 1e-10,
            "i22={}",
            e.information[(1, 1)]
        );
        assert!(
            (e.information[(2, 2)] - 200.0).abs() < 1e-10,
            "i33={}",
            e.information[(2, 2)]
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_information_matrix() -> TestResult {
        // EDGE_SE3:QUAT: from to tx ty tz qx qy qz qw + 21 upper-triangular values
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 0 0.0 0.0 0.0 0.0 0.0 0.0 1.0")?;
        writeln!(f, "VERTEX_SE3:QUAT 1 1.0 0.0 0.0 0.0 0.0 0.0 1.0")?;
        // Information matrix: identity (diagonal 1.0, off-diagonal 0.0), upper triangular = 21 values
        let info_vals = "100.0 0.0 0.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 100.0 0.0 100.0";
        writeln!(
            f,
            "EDGE_SE3:QUAT 0 1 1.0 0.0 0.0 0.0 0.0 0.0 1.0 {}",
            info_vals
        )?;
        f.flush()?;
        let graph = G2oLoader::load(f.path())?;
        assert_eq!(graph.edges_se3.len(), 1);
        let e = &graph.edges_se3[0];
        assert!((e.information[(0, 0)] - 100.0).abs() < 1e-10);
        assert!((e.information[(1, 1)] - 100.0).abs() < 1e-10);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_vertex_se2 error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_vertex_se2_invalid_x() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE2 0 bad 2.0 0.5")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid x in VERTEX_SE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se2_invalid_y() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE2 0 1.0 bad 0.5")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid y in VERTEX_SE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se2_invalid_theta() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE2 0 1.0 2.0 bad")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid theta in VERTEX_SE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se2_missing_fields() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE2 0 1.0")?; // only 3 parts (needs 5)
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::MissingFields { .. })),
            "VERTEX_SE2 with too few fields should return MissingFields"
        );
        Ok(())
    }

    #[test]
    fn test_parse_duplicate_vertex_se2() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE2 3 1.0 2.0 0.0")?;
        writeln!(f, "VERTEX_SE2 3 3.0 4.0 0.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::DuplicateVertex { id: 3 })),
            "duplicate VERTEX_SE2 ID should return DuplicateVertex"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_vertex_se3 error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_vertex_se3_missing_fields() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 0 1.0 2.0")?; // only 4 parts (needs 9)
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::MissingFields { .. })),
            "VERTEX_SE3:QUAT with too few fields should return MissingFields"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se3_invalid_translation() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 0 bad 2.0 3.0 0.0 0.0 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid translation in VERTEX_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se3_invalid_quaternion_field() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 0 1.0 2.0 3.0 bad 0.0 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid quaternion field should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_duplicate_vertex_se3() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 7 1.0 0.0 0.0 0.0 0.0 0.0 1.0")?;
        writeln!(f, "VERTEX_SE3:QUAT 7 2.0 0.0 0.0 0.0 0.0 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::DuplicateVertex { id: 7 })),
            "duplicate VERTEX_SE3:QUAT ID should return DuplicateVertex"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se3_invalid_id() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT bad 1.0 2.0 3.0 0.0 0.0 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid id in VERTEX_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se3_invalid_y() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 0 1.0 bad 3.0 0.0 0.0 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid y in VERTEX_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se3_invalid_z() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 0 1.0 2.0 bad 0.0 0.0 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid z in VERTEX_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se3_invalid_qy() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 0 1.0 2.0 3.0 0.0 bad 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid qy in VERTEX_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se3_invalid_qz() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 0 1.0 2.0 3.0 0.0 0.0 bad 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid qz in VERTEX_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex_se3_invalid_qw() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "VERTEX_SE3:QUAT 0 1.0 2.0 3.0 0.0 0.0 0.0 bad")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid qw in VERTEX_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_edge_se2 error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_edge_se2_missing_fields() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "EDGE_SE2 0 1 1.0 0.0")?; // only 5 parts (needs 12)
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::MissingFields { .. })),
            "EDGE_SE2 with too few fields should return MissingFields"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se2_invalid_from_id() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(
            f,
            "EDGE_SE2 bad 1 1.0 0.0 0.0 500.0 0.0 0.0 500.0 0.0 200.0"
        )?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid from-ID in EDGE_SE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se2_invalid_measurement() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "EDGE_SE2 0 1 bad 0.0 0.0 500.0 0.0 0.0 500.0 0.0 200.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            result.is_err(),
            "invalid measurement in EDGE_SE2 should return error"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se2_invalid_to_id() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(
            f,
            "EDGE_SE2 0 bad 1.0 0.0 0.0 500.0 0.0 0.0 500.0 0.0 200.0"
        )?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid to-ID in EDGE_SE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se2_invalid_dy() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "EDGE_SE2 0 1 1.0 bad 0.0 500.0 0.0 0.0 500.0 0.0 200.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            result.is_err(),
            "invalid dy in EDGE_SE2 should return error"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se2_invalid_dtheta() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "EDGE_SE2 0 1 1.0 0.0 bad 500.0 0.0 0.0 500.0 0.0 200.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            result.is_err(),
            "invalid dtheta in EDGE_SE2 should return error"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se2_invalid_info_matrix() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "EDGE_SE2 0 1 1.0 0.0 0.0 bad 0.0 0.0 500.0 0.0 200.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            result.is_err(),
            "invalid info-matrix value in EDGE_SE2 should return error"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_edge_se3 error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_edge_se3_missing_fields() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "EDGE_SE3:QUAT 0 1 1.0 0.0")?; // only 5 parts (needs >= 10)
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::MissingFields { .. })),
            "EDGE_SE3:QUAT with too few fields should return MissingFields"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_invalid_translation() -> TestResult {
        let mut f = NamedTempFile::new()?;
        // bad tx
        writeln!(f, "EDGE_SE3:QUAT 0 1 bad 0.0 0.0 0.0 0.0 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid translation in EDGE_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_invalid_quaternion_field() -> TestResult {
        let mut f = NamedTempFile::new()?;
        // bad qz field
        let info_vals =
            "1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0";
        writeln!(
            f,
            "EDGE_SE3:QUAT 0 1 1.0 0.0 0.0 0.0 0.0 bad 1.0 {}",
            info_vals
        )?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid quaternion field in EDGE_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_invalid_from_id() -> TestResult {
        let mut f = NamedTempFile::new()?;
        let info_vals =
            "1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0";
        writeln!(
            f,
            "EDGE_SE3:QUAT bad 1 1.0 0.0 0.0 0.0 0.0 0.0 1.0 {}",
            info_vals
        )?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid from-id in EDGE_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_invalid_to_id() -> TestResult {
        let mut f = NamedTempFile::new()?;
        let info_vals =
            "1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0";
        writeln!(
            f,
            "EDGE_SE3:QUAT 0 bad 1.0 0.0 0.0 0.0 0.0 0.0 1.0 {}",
            info_vals
        )?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid to-id in EDGE_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_invalid_ty() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "EDGE_SE3:QUAT 0 1 1.0 bad 0.0 0.0 0.0 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid ty in EDGE_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_invalid_tz() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "EDGE_SE3:QUAT 0 1 1.0 0.0 bad 0.0 0.0 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid tz in EDGE_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_invalid_qx() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "EDGE_SE3:QUAT 0 1 1.0 0.0 0.0 bad 0.0 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid qx in EDGE_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_invalid_qy() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "EDGE_SE3:QUAT 0 1 1.0 0.0 0.0 0.0 bad 0.0 1.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid qy in EDGE_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_invalid_qw() -> TestResult {
        let mut f = NamedTempFile::new()?;
        let info_vals =
            "1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0";
        writeln!(
            f,
            "EDGE_SE3:QUAT 0 1 1.0 0.0 0.0 0.0 0.0 0.0 bad {}",
            info_vals
        )?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid qw in EDGE_SE3:QUAT should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge_se3_invalid_info_matrix() -> TestResult {
        let mut f = NamedTempFile::new()?;
        let info_vals = "bad 0.0 0.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 100.0 0.0 100.0";
        writeln!(
            f,
            "EDGE_SE3:QUAT 0 1 1.0 0.0 0.0 0.0 0.0 0.0 1.0 {}",
            info_vals
        )?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            result.is_err(),
            "invalid info-matrix values in EDGE_SE3:QUAT should return error"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_line: comment / empty / unknown token paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_comment_and_empty_lines_ignored() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "# This is a comment")?;
        writeln!(f, "VERTEX_SE2 0 1.0 2.0 0.0")?;
        writeln!(f)?; // empty line
        writeln!(f, "VERTEX_SE2 1 3.0 4.0 0.0")?;
        f.flush()?;
        let graph = G2oLoader::load(f.path())?;
        assert_eq!(
            graph.vertices_se2.len(),
            2,
            "comments and empty lines should be ignored"
        );
        Ok(())
    }

    #[test]
    fn test_parse_unknown_token_ignored() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "FIX 0")?; // unknown token
        writeln!(f, "VERTEX_SE2 0 0.0 0.0 0.0")?;
        f.flush()?;
        let graph = G2oLoader::load(f.path())?;
        assert_eq!(graph.vertices_se2.len(), 1);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_parallel (> 1000 lines triggers parallel path)
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_parallel_large_se2_file() -> TestResult {
        let mut f = NamedTempFile::new()?;
        for i in 0..1001usize {
            writeln!(f, "VERTEX_SE2 {} {} {} 0.0", i, i as f64, i as f64)?;
        }
        f.flush()?;
        let graph = G2oLoader::load(f.path())?;
        assert_eq!(graph.vertices_se2.len(), 1001);
        Ok(())
    }

    #[test]
    fn test_parse_parallel_with_se2_edges() -> TestResult {
        let mut f = NamedTempFile::new()?;
        for i in 0..1001usize {
            writeln!(f, "VERTEX_SE2 {} 0.0 0.0 0.0", i)?;
        }
        writeln!(f, "EDGE_SE2 0 1 1.0 0.0 0.0 500.0 0.0 0.0 500.0 0.0 200.0")?;
        f.flush()?;
        let graph = G2oLoader::load(f.path())?;
        assert_eq!(graph.vertices_se2.len(), 1001);
        assert_eq!(graph.edges_se2.len(), 1);
        Ok(())
    }

    #[test]
    fn test_parse_parallel_with_se3_vertices() -> TestResult {
        let mut f = NamedTempFile::new()?;
        for i in 0..1001usize {
            writeln!(
                f,
                "VERTEX_SE3:QUAT {} 0.0 0.0 {} 0.0 0.0 0.0 1.0",
                i, i as f64
            )?;
        }
        f.flush()?;
        let graph = G2oLoader::load(f.path())?;
        assert_eq!(graph.vertices_se3.len(), 1001);
        Ok(())
    }

    #[test]
    fn test_parse_parallel_with_se3_edges() -> TestResult {
        let mut f = NamedTempFile::new()?;
        for i in 0..1001usize {
            writeln!(f, "VERTEX_SE3:QUAT {} 0.0 0.0 0.0 0.0 0.0 0.0 1.0", i)?;
        }
        let info_vals =
            "1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0";
        writeln!(
            f,
            "EDGE_SE3:QUAT 0 1 1.0 0.0 0.0 0.0 0.0 0.0 1.0 {}",
            info_vals
        )?;
        f.flush()?;
        let graph = G2oLoader::load(f.path())?;
        assert_eq!(graph.vertices_se3.len(), 1001);
        assert_eq!(graph.edges_se3.len(), 1);
        Ok(())
    }

    #[test]
    fn test_parse_parallel_duplicate_vertex_returns_error() -> TestResult {
        let mut f = NamedTempFile::new()?;
        for i in 0..1000usize {
            writeln!(f, "VERTEX_SE2 {} 0.0 0.0 0.0", i)?;
        }
        writeln!(f, "VERTEX_SE2 0 9.0 9.0 0.0")?; // duplicate of ID 0
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::DuplicateVertex { id: 0 })),
            "duplicate vertex in parallel parse should return DuplicateVertex"
        );
        Ok(())
    }

    #[test]
    fn test_parse_parallel_comment_and_empty_lines_ignored() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "# parallel parse comment test")?;
        for i in 0..1000usize {
            writeln!(f, "VERTEX_SE2 {} 0.0 0.0 0.0", i)?;
        }
        writeln!(f)?; // empty line at end
        f.flush()?;
        let graph = G2oLoader::load(f.path())?;
        assert_eq!(graph.vertices_se2.len(), 1000);
        Ok(())
    }
}
