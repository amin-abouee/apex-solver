use crate::{EdgeSE2, Graph, GraphLoader, IoError, VertexSE2};
use memmap2::Mmap;
use std::{fs, io::Write, path::Path};

/// TORO format loader
pub struct ToroLoader;

impl GraphLoader for ToroLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<Graph, IoError> {
        let path_ref = path.as_ref();
        let file = fs::File::open(path_ref).map_err(|e| {
            IoError::Io(e).log_with_source(format!("Failed to open TORO file: {:?}", path_ref))
        })?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                IoError::Io(e)
                    .log_with_source(format!("Failed to memory-map TORO file: {:?}", path_ref))
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
        // TORO only supports SE2
        if !graph.vertices_se3.is_empty() || !graph.edges_se3.is_empty() {
            return Err(IoError::UnsupportedFormat(
                "TORO format only supports SE2 (2D) graphs. Use G2O format for SE3 data."
                    .to_string(),
            )
            .log());
        }

        let path_ref = path.as_ref();
        let mut file = fs::File::create(path_ref).map_err(|e| {
            IoError::Io(e).log_with_source(format!("Failed to create TORO file: {:?}", path_ref))
        })?;

        // Write SE2 vertices (sorted by ID)
        let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().collect();
        vertex_ids.sort();

        for id in vertex_ids {
            let vertex = &graph.vertices_se2[id];
            writeln!(
                file,
                "VERTEX2 {} {:.17e} {:.17e} {:.17e}",
                vertex.id,
                vertex.x(),
                vertex.y(),
                vertex.theta()
            )
            .map_err(|e| {
                IoError::Io(e).log_with_source(format!("Failed to write TORO vertex {}", vertex.id))
            })?;
        }

        // Write SE2 edges
        // TORO format: EDGE2 <id1> <id2> <dx> <dy> <dtheta> <i11> <i12> <i22> <i33> <i13> <i23>
        for edge in &graph.edges_se2 {
            let meas = &edge.measurement;
            let info = &edge.information;

            writeln!(
                file,
                "EDGE2 {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                edge.from,
                edge.to,
                meas.x(),
                meas.y(),
                meas.angle(),
                info[(0, 0)], // i11
                info[(0, 1)], // i12
                info[(1, 1)], // i22
                info[(2, 2)], // i33
                info[(0, 2)], // i13
                info[(1, 2)]  // i23
            )
            .map_err(|e| {
                IoError::Io(e).log_with_source(format!(
                    "Failed to write TORO edge {} -> {}",
                    edge.from, edge.to
                ))
            })?;
        }

        Ok(())
    }
}

impl ToroLoader {
    fn parse_content(content: &str) -> Result<Graph, IoError> {
        let lines: Vec<&str> = content.lines().collect();
        let mut graph = Graph::new();

        for (line_num, line) in lines.iter().enumerate() {
            Self::parse_line(line, line_num + 1, &mut graph)?;
        }

        Ok(graph)
    }

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
            "VERTEX2" => {
                let vertex = Self::parse_vertex2(&parts, line_num)?;
                let id = vertex.id;
                if graph.vertices_se2.insert(id, vertex).is_some() {
                    return Err(IoError::DuplicateVertex { id });
                }
            }
            "EDGE2" => {
                let edge = Self::parse_edge2(&parts, line_num)?;
                graph.edges_se2.push(edge);
            }
            _ => {
                // Skip unknown types silently for compatibility
            }
        }

        Ok(())
    }

    fn parse_vertex2(parts: &[&str], line_num: usize) -> Result<VertexSE2, IoError> {
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

    fn parse_edge2(parts: &[&str], line_num: usize) -> Result<EdgeSE2, IoError> {
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

        // Parse TORO information matrix (I11, I12, I22, I33, I13, I23)
        let i11 = parts[6]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[6].to_string(),
            })?;
        let i12 = parts[7]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[7].to_string(),
            })?;
        let i22 = parts[8]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[8].to_string(),
            })?;
        let i33 = parts[9]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[9].to_string(),
            })?;
        let i13 = parts[10]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[10].to_string(),
            })?;
        let i23 = parts[11]
            .parse::<f64>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[11].to_string(),
            })?;

        let information = nalgebra::Matrix3::new(i11, i12, i13, i12, i22, i23, i13, i23, i33);

        Ok(EdgeSE2::new(from, to, dx, dy, dtheta, information))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EdgeSE3, VertexSE3};
    use nalgebra::{Matrix3, UnitQuaternion, Vector3};
    use std::io::Write;
    use tempfile::NamedTempFile;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn write_toro_content(content: &str) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
        let mut f = NamedTempFile::new()?;
        write!(f, "{}", content)?;
        f.flush()?;
        Ok(f)
    }

    #[test]
    fn test_parse_vertex2_and_edge2() -> TestResult {
        let content = "VERTEX2 0 1.0 2.0 0.5\n\
                       VERTEX2 1 3.0 4.0 1.0\n\
                       EDGE2 0 1 0.5 0.3 0.1 500.0 0.0 500.0 200.0 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let graph = ToroLoader::load(f.path())?;
        assert_eq!(graph.vertices_se2.len(), 2);
        assert_eq!(graph.edges_se2.len(), 1);
        let v0 = &graph.vertices_se2[&0];
        assert!((v0.x() - 1.0).abs() < 1e-10);
        assert!((v0.y() - 2.0).abs() < 1e-10);
        let e = &graph.edges_se2[0];
        assert_eq!(e.from, 0);
        assert_eq!(e.to, 1);
        assert!((e.information[(0, 0)] - 500.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_write_and_reload_round_trip() -> TestResult {
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
        ToroLoader::write(&graph, f.path())?;
        let loaded = ToroLoader::load(f.path())?;

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
    fn test_write_rejects_se3_vertices() -> TestResult {
        let mut graph = Graph::new();
        graph.vertices_se3.insert(
            0,
            VertexSE3::new(0, Vector3::zeros(), UnitQuaternion::identity()),
        );
        let f = NamedTempFile::new()?;
        let result = ToroLoader::write(&graph, f.path());
        assert!(
            matches!(result, Err(IoError::UnsupportedFormat(_))),
            "should reject graph with SE3 vertices"
        );
        Ok(())
    }

    #[test]
    fn test_write_rejects_se3_edges() -> TestResult {
        let mut graph = Graph::new();
        graph.edges_se3.push(EdgeSE3::new(
            0,
            1,
            Vector3::zeros(),
            UnitQuaternion::identity(),
            nalgebra::Matrix6::identity(),
        ));
        let f = NamedTempFile::new()?;
        let result = ToroLoader::write(&graph, f.path());
        assert!(
            matches!(result, Err(IoError::UnsupportedFormat(_))),
            "should reject graph with SE3 edges"
        );
        Ok(())
    }

    #[test]
    fn test_duplicate_vertex_returns_error() -> TestResult {
        let content = "VERTEX2 5 1.0 2.0 0.0\nVERTEX2 5 3.0 4.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::DuplicateVertex { id: 5 })),
            "duplicate vertex ID should return DuplicateVertex error"
        );
        Ok(())
    }

    #[test]
    fn test_parse_missing_vertex_fields() -> TestResult {
        // VERTEX2 needs 5 fields: VERTEX2 id x y theta
        let content = "VERTEX2 0 1.0\n"; // only 3 fields
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(result.is_err(), "VERTEX2 with too few fields should fail");
        Ok(())
    }

    #[test]
    fn test_parse_missing_edge_fields() -> TestResult {
        // EDGE2 needs 12 fields
        let content = "EDGE2 0 1 0.5 0.3\n"; // only 5 fields
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(result.is_err(), "EDGE2 with too few fields should fail");
        Ok(())
    }

    #[test]
    fn test_comment_and_empty_lines_ignored() -> TestResult {
        let content = "# this is a comment\n\
                       VERTEX2 0 1.0 2.0 0.0\n\
                       \n\
                       VERTEX2 1 2.0 3.0 0.0\n";
        let f = write_toro_content(content)?;
        let graph = ToroLoader::load(f.path())?;
        assert_eq!(
            graph.vertices_se2.len(),
            2,
            "comments and blank lines should be ignored"
        );
        Ok(())
    }

    #[test]
    fn test_unknown_token_ignored() -> TestResult {
        let content = "UNKNOWN_TOKEN 1 2 3\nVERTEX2 0 0.0 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let graph = ToroLoader::load(f.path())?;
        assert_eq!(
            graph.vertices_se2.len(),
            1,
            "unknown token lines should be silently skipped"
        );
        Ok(())
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = ToroLoader::load("/no/such/file.graph");
        assert!(result.is_err(), "loading missing file should return Err");
    }

    #[test]
    fn test_write_empty_graph() -> TestResult {
        let graph = Graph::new();
        let f = NamedTempFile::new()?;
        ToroLoader::write(&graph, f.path())?;
        let loaded = ToroLoader::load(f.path())?;
        assert_eq!(loaded.vertices_se2.len(), 0);
        assert_eq!(loaded.edges_se2.len(), 0);
        Ok(())
    }

    #[test]
    fn test_parse_vertex2_invalid_number() -> TestResult {
        let content = "VERTEX2 0 bad 2.0 0.0\n"; // bad x value
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(result.is_err(), "invalid number in VERTEX2 should fail");
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_vertex2 additional error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_vertex2_invalid_id() -> TestResult {
        let content = "VERTEX2 bad 1.0 2.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid id in VERTEX2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex2_invalid_y() -> TestResult {
        let content = "VERTEX2 0 1.0 bad 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid y in VERTEX2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_vertex2_invalid_theta() -> TestResult {
        let content = "VERTEX2 0 1.0 2.0 bad\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid theta in VERTEX2 should return InvalidNumber"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_edge2 error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_edge2_invalid_from() -> TestResult {
        let content = "EDGE2 bad 1 0.5 0.3 0.1 500.0 0.0 500.0 200.0 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid from-id in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge2_invalid_to() -> TestResult {
        let content = "EDGE2 0 bad 0.5 0.3 0.1 500.0 0.0 500.0 200.0 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid to-id in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge2_invalid_dx() -> TestResult {
        let content = "EDGE2 0 1 bad 0.3 0.1 500.0 0.0 500.0 200.0 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid dx in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge2_invalid_dy() -> TestResult {
        let content = "EDGE2 0 1 0.5 bad 0.1 500.0 0.0 500.0 200.0 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid dy in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge2_invalid_dtheta() -> TestResult {
        let content = "EDGE2 0 1 0.5 0.3 bad 500.0 0.0 500.0 200.0 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid dtheta in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge2_invalid_i11() -> TestResult {
        let content = "EDGE2 0 1 0.5 0.3 0.1 bad 0.0 500.0 200.0 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid i11 in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge2_invalid_i12() -> TestResult {
        let content = "EDGE2 0 1 0.5 0.3 0.1 500.0 bad 500.0 200.0 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid i12 in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge2_invalid_i22() -> TestResult {
        let content = "EDGE2 0 1 0.5 0.3 0.1 500.0 0.0 bad 200.0 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid i22 in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge2_invalid_i33() -> TestResult {
        let content = "EDGE2 0 1 0.5 0.3 0.1 500.0 0.0 500.0 bad 0.0 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid i33 in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge2_invalid_i13() -> TestResult {
        let content = "EDGE2 0 1 0.5 0.3 0.1 500.0 0.0 500.0 200.0 bad 0.0\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid i13 in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_parse_edge2_invalid_i23() -> TestResult {
        let content = "EDGE2 0 1 0.5 0.3 0.1 500.0 0.0 500.0 200.0 0.0 bad\n";
        let f = write_toro_content(content)?;
        let result = ToroLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid i23 in EDGE2 should return InvalidNumber"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Round-trip fidelity
    // -------------------------------------------------------------------------

    #[test]
    fn test_edge_measurement_all_components_round_trip() -> TestResult {
        let mut graph = Graph::new();
        graph
            .vertices_se2
            .insert(0, VertexSE2::new(0, 0.0, 0.0, 0.0));
        graph
            .vertices_se2
            .insert(1, VertexSE2::new(1, 1.0, 0.0, 0.0));
        let info = Matrix3::identity();
        graph
            .edges_se2
            .push(EdgeSE2::new(0, 1, 1.5, 2.5, 0.7, info));

        let f = NamedTempFile::new()?;
        ToroLoader::write(&graph, f.path())?;
        let loaded = ToroLoader::load(f.path())?;

        let e = &loaded.edges_se2[0];
        assert!((e.measurement.x() - 1.5).abs() < 1e-10, "dx mismatch");
        assert!((e.measurement.y() - 2.5).abs() < 1e-10, "dy mismatch");
        assert!(
            (e.measurement.angle() - 0.7).abs() < 1e-10,
            "dtheta mismatch"
        );
        Ok(())
    }

    #[test]
    fn test_off_diagonal_info_matrix_round_trip() -> TestResult {
        let mut graph = Graph::new();
        graph
            .vertices_se2
            .insert(0, VertexSE2::new(0, 0.0, 0.0, 0.0));
        graph
            .vertices_se2
            .insert(1, VertexSE2::new(1, 1.0, 0.0, 0.0));
        // Symmetric matrix with off-diagonal entries
        let info = Matrix3::new(500.0, 10.0, 5.0, 10.0, 400.0, 3.0, 5.0, 3.0, 200.0);
        graph
            .edges_se2
            .push(EdgeSE2::new(0, 1, 1.0, 0.0, 0.0, info));

        let f = NamedTempFile::new()?;
        ToroLoader::write(&graph, f.path())?;
        let loaded = ToroLoader::load(f.path())?;

        let e = &loaded.edges_se2[0];
        assert!((e.information[(0, 0)] - 500.0).abs() < 1e-10, "i11");
        assert!((e.information[(0, 1)] - 10.0).abs() < 1e-10, "i12");
        assert!((e.information[(1, 1)] - 400.0).abs() < 1e-10, "i22");
        assert!((e.information[(2, 2)] - 200.0).abs() < 1e-10, "i33");
        assert!((e.information[(0, 2)] - 5.0).abs() < 1e-10, "i13");
        assert!((e.information[(1, 2)] - 3.0).abs() < 1e-10, "i23");
        Ok(())
    }

    #[test]
    fn test_multiple_edges_round_trip() -> TestResult {
        let mut graph = Graph::new();
        for i in 0..4usize {
            graph
                .vertices_se2
                .insert(i, VertexSE2::new(i, i as f64, 0.0, 0.0));
        }
        let info = Matrix3::identity();
        for i in 0..3usize {
            graph
                .edges_se2
                .push(EdgeSE2::new(i, i + 1, 1.0, 0.0, 0.0, info));
        }

        let f = NamedTempFile::new()?;
        ToroLoader::write(&graph, f.path())?;
        let loaded = ToroLoader::load(f.path())?;

        assert_eq!(loaded.vertices_se2.len(), 4);
        assert_eq!(loaded.edges_se2.len(), 3);
        Ok(())
    }

    #[test]
    fn test_vertex_theta_preserved_round_trip() -> TestResult {
        let mut graph = Graph::new();
        graph
            .vertices_se2
            .insert(0, VertexSE2::new(0, 1.0, 2.0, std::f64::consts::PI / 4.0));

        let f = NamedTempFile::new()?;
        ToroLoader::write(&graph, f.path())?;
        let loaded = ToroLoader::load(f.path())?;

        let v = &loaded.vertices_se2[&0];
        assert!(
            (v.theta() - std::f64::consts::PI / 4.0).abs() < 1e-10,
            "theta not preserved"
        );
        Ok(())
    }

    #[test]
    fn test_load_invalid_utf8_returns_err() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0xFF, 0xFE, 0x80, 0x00, 0xAB]).unwrap();
        let result = ToroLoader::load(f.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_write_to_nonexistent_dir_returns_err() {
        let mut graph = Graph::new();
        graph.vertices_se2.insert(0, VertexSE2::new(0, 0.0, 0.0, 0.0));
        let result = ToroLoader::write(&graph, "/nonexistent_dir_xyz/output.toro");
        assert!(result.is_err());
    }
}
