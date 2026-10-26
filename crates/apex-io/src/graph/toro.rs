use super::error::{parse_field, require_fields};
use super::{EdgeSE2, Graph, GraphError, GraphLoader, GraphResult, VertexSE2};
use apex_error::ErrorLogging;
use std::collections::BTreeMap;
use std::{fs, io::Write, path::Path};
use tracing::warn;

/// TORO format loader
pub struct ToroLoader;

impl GraphLoader for ToroLoader {
    fn load<P: AsRef<Path>>(path: P) -> GraphResult<Graph> {
        let path_ref = path.as_ref();
        let content =
            fs::read_to_string(path_ref).map_err(|e| GraphError::read(path_ref, e).log())?;

        Self::parse_content(&content)
    }

    fn write<P: AsRef<Path>>(graph: &Graph, path: P) -> GraphResult<()> {
        if !graph.vertices_se3.is_empty() || !graph.edges_se3.is_empty() {
            return Err(GraphError::ToroSe3 {
                vertices: graph.vertices_se3.len(),
                edges: graph.edges_se3.len(),
            }
            .log());
        }

        let path_ref = path.as_ref();
        Self::write_to(graph, path_ref).map_err(|e| GraphError::write(path_ref, e).log())
    }
}

impl ToroLoader {
    fn write_to(graph: &Graph, path: &Path) -> std::io::Result<()> {
        let mut file = std::io::BufWriter::new(fs::File::create(path)?);

        let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().collect();
        vertex_ids.sort();
        for id in vertex_ids {
            let Some(vertex) = graph.vertices_se2.get(id) else {
                continue;
            };
            writeln!(
                file,
                "VERTEX2 {} {:.17e} {:.17e} {:.17e}",
                vertex.id,
                vertex.x(),
                vertex.y(),
                vertex.theta()
            )?;
        }

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
            )?;
        }

        file.flush()
    }

    /// Parse TORO text. A line error is logged here, once, where the load stops.
    fn parse_content(content: &str) -> GraphResult<Graph> {
        let mut graph = Graph::new();
        let mut skipped: BTreeMap<&str, usize> = BTreeMap::new();

        for (index, line) in content.lines().enumerate() {
            Self::parse_line(line, index + 1, &mut graph, &mut skipped).map_err(|e| e.log())?;
        }

        for (record, count) in skipped {
            warn!(
                record,
                count, "skipped TORO records of a type this loader does not read"
            );
        }

        Ok(graph)
    }

    fn parse_line<'a>(
        line: &'a str,
        line_num: usize,
        graph: &mut Graph,
        skipped: &mut BTreeMap<&'a str, usize>,
    ) -> GraphResult<()> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(());
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        let Some(&tag) = parts.first() else {
            return Ok(());
        };

        match tag {
            "VERTEX2" => {
                let vertex = Self::parse_vertex2(&parts, line_num)?;
                let id = vertex.id;
                if graph.vertices_se2.insert(id, vertex).is_some() {
                    return Err(GraphError::DuplicateVertex { id });
                }
            }
            "EDGE2" => {
                let edge = Self::parse_edge2(&parts, line_num)?;
                graph.edges_se2.push(edge);
            }
            other => *skipped.entry(other).or_insert(0) += 1,
        }

        Ok(())
    }

    fn parse_vertex2(parts: &[&str], line_num: usize) -> GraphResult<VertexSE2> {
        const RECORD: &str = "VERTEX2";
        require_fields(parts, 5, line_num, RECORD)?;
        let id = parse_field::<usize>(parts, 1, line_num, RECORD)?;
        let x = parse_field::<f64>(parts, 2, line_num, RECORD)?;
        let y = parse_field::<f64>(parts, 3, line_num, RECORD)?;
        let theta = parse_field::<f64>(parts, 4, line_num, RECORD)?;
        Ok(VertexSE2::new(id, x, y, theta))
    }

    fn parse_edge2(parts: &[&str], line_num: usize) -> GraphResult<EdgeSE2> {
        const RECORD: &str = "EDGE2";
        require_fields(parts, 12, line_num, RECORD)?;
        let from = parse_field::<usize>(parts, 1, line_num, RECORD)?;
        let to = parse_field::<usize>(parts, 2, line_num, RECORD)?;
        // Measurement (dx, dy, dtheta), then TORO's information order:
        // I11, I12, I22, I33, I13, I23.
        let mut values = [0.0; 9];
        for (offset, value) in values.iter_mut().enumerate() {
            *value = parse_field::<f64>(parts, 3 + offset, line_num, RECORD)?;
        }
        let [dx, dy, dtheta, i11, i12, i22, i33, i13, i23] = values;

        let information = nalgebra::Matrix3::new(i11, i12, i13, i12, i22, i23, i13, i23, i33);

        Ok(EdgeSE2::new(from, to, dx, dy, dtheta, information))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeSE3, VertexSE3};
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
            matches!(result, Err(GraphError::ToroSe3 { .. })),
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
            matches!(result, Err(GraphError::ToroSe3 { .. })),
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
            matches!(result, Err(GraphError::DuplicateVertex { id: 5 })),
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
    fn test_load_nonexistent_file() -> TestResult {
        let result = ToroLoader::load("/no/such/file.graph");
        assert!(result.is_err(), "loading missing file should return Err");
        Ok(())
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
    fn test_load_invalid_utf8_returns_err() -> TestResult {
        let mut f = NamedTempFile::new()?;
        f.write_all(&[0xFF, 0xFE, 0x80, 0x00, 0xAB])?;
        let result = ToroLoader::load(f.path());
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_write_to_nonexistent_dir_returns_err() -> TestResult {
        let mut graph = Graph::new();
        graph
            .vertices_se2
            .insert(0, VertexSE2::new(0, 0.0, 0.0, 0.0));
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("nested").join("deep").join("output.toro");
        let result = ToroLoader::write(&graph, &path);
        assert!(result.is_err());
        Ok(())
    }
}
