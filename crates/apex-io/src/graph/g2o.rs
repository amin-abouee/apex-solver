use super::error::{parse_field, require_fields};
use super::{EdgeSE2, EdgeSE3, Graph, GraphError, GraphLoader, GraphResult, VertexSE2, VertexSE3};
use apex_error::ErrorLogging;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::{fs::File, io::Write, path::Path};
use tracing::warn;

/// Files with more lines than this are parsed in parallel.
const MIN_LINES_FOR_PARALLEL: usize = 1000;

/// High-performance G2O file loader
pub struct G2oLoader;

impl GraphLoader for G2oLoader {
    fn load<P: AsRef<Path>>(path: P) -> GraphResult<Graph> {
        let path_ref = path.as_ref();
        let content =
            std::fs::read_to_string(path_ref).map_err(|e| GraphError::read(path_ref, e).log())?;

        Self::parse_content(&content)
    }

    fn write<P: AsRef<Path>>(graph: &Graph, path: P) -> GraphResult<()> {
        let path_ref = path.as_ref();
        Self::write_to(graph, path_ref).map_err(|e| GraphError::write(path_ref, e).log())
    }
}

impl G2oLoader {
    fn write_to(graph: &Graph, path: &Path) -> std::io::Result<()> {
        let mut file = std::io::BufWriter::new(File::create(path)?);

        writeln!(file, "# G2O file written by Apex Solver")?;
        writeln!(
            file,
            "# Timestamp: {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        )?;
        writeln!(
            file,
            "# SE2 vertices: {}, SE3 vertices: {}, SE2 edges: {}, SE3 edges: {}",
            graph.vertices_se2.len(),
            graph.vertices_se3.len(),
            graph.edges_se2.len(),
            graph.edges_se3.len()
        )?;
        writeln!(file)?;

        let mut se2_ids: Vec<_> = graph.vertices_se2.keys().collect();
        se2_ids.sort();
        for id in se2_ids {
            let Some(vertex) = graph.vertices_se2.get(id) else {
                continue;
            };
            writeln!(
                file,
                "VERTEX_SE2 {} {:.17e} {:.17e} {:.17e}",
                vertex.id,
                vertex.x(),
                vertex.y(),
                vertex.theta()
            )?;
        }

        let mut se3_ids: Vec<_> = graph.vertices_se3.keys().collect();
        se3_ids.sort();
        for id in se3_ids {
            let Some(vertex) = graph.vertices_se3.get(id) else {
                continue;
            };
            let trans = vertex.translation();
            let quat = vertex.rotation();
            writeln!(
                file,
                "VERTEX_SE3:QUAT {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                vertex.id, trans.x, trans.y, trans.z, quat.i, quat.j, quat.k, quat.w
            )?;
        }

        for edge in &graph.edges_se2 {
            let meas = &edge.measurement;
            let info = &edge.information;

            // g2o's EdgeSE2 information order is the upper triangle row by
            // row, I11 I12 I13 I22 I23 I33, which is what the reader expects
            // (TORO's EDGE2 is the one ordered I11 I12 I22 I33 I13 I23).
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
                info[(0, 2)],
                info[(1, 1)],
                info[(1, 2)],
                info[(2, 2)]
            )?;
        }

        for edge in &graph.edges_se3 {
            let trans = edge.measurement.translation();
            let quat = edge.measurement.rotation_quaternion();
            let info = &edge.information;

            write!(
                file,
                "EDGE_SE3:QUAT {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                edge.from, edge.to, trans.x, trans.y, trans.z, quat.i, quat.j, quat.k, quat.w
            )?;
            // Upper triangle of the 6x6 information matrix, row by row (21 values).
            for i in 0..6 {
                for j in i..6 {
                    write!(file, " {:.17e}", info[(i, j)])?;
                }
            }
            writeln!(file)?;
        }

        file.flush()
    }

    /// Parse G2O text; files above [`MIN_LINES_FOR_PARALLEL`] lines parse in parallel.
    ///
    /// The per-line parsers build their errors without logging; the first
    /// error found is logged here, once, where the load is aborted.
    fn parse_content(content: &str) -> GraphResult<Graph> {
        let lines: Vec<&str> = content.lines().collect();
        fn parse<'a>((index, line): (usize, &&'a str)) -> GraphResult<Option<Record<'a>>> {
            G2oLoader::parse_record(line, index + 1)
        }
        let records: GraphResult<Vec<Option<Record<'_>>>> = if lines.len() > MIN_LINES_FOR_PARALLEL
        {
            lines.par_iter().enumerate().map(parse).collect()
        } else {
            lines.iter().enumerate().map(parse).collect()
        };
        let records = records.map_err(|e| e.log())?;

        let estimated_vertices = lines.len() / 4;
        let estimated_edges = estimated_vertices * 3;
        let mut graph = Graph {
            vertices_se2: HashMap::with_capacity(estimated_vertices),
            vertices_se3: HashMap::with_capacity(estimated_vertices),
            edges_se2: Vec::with_capacity(estimated_edges),
            edges_se3: Vec::with_capacity(estimated_edges),
        };
        let mut skipped: BTreeMap<&str, usize> = BTreeMap::new();

        for record in records.into_iter().flatten() {
            match record {
                Record::VertexSE2(vertex) => {
                    let id = vertex.id;
                    if graph.vertices_se2.insert(id, vertex).is_some() {
                        return Err(GraphError::DuplicateVertex { id }.log());
                    }
                }
                Record::VertexSE3(vertex) => {
                    let id = vertex.id;
                    if graph.vertices_se3.insert(id, vertex).is_some() {
                        return Err(GraphError::DuplicateVertex { id }.log());
                    }
                }
                Record::EdgeSE2(edge) => graph.edges_se2.push(edge),
                Record::EdgeSE3(edge) => graph.edges_se3.push(*edge),
                Record::Unknown(tag) => *skipped.entry(tag).or_insert(0) += 1,
            }
        }

        for (record, count) in skipped {
            warn!(
                record,
                count, "skipped g2o records of a type this loader does not read"
            );
        }

        Ok(graph)
    }

    /// Parse one line; comments and blank lines give `None`.
    fn parse_record(line: &str, line_num: usize) -> GraphResult<Option<Record<'_>>> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        let Some(&tag) = parts.first() else {
            return Ok(None);
        };

        let record = match tag {
            "VERTEX_SE2" => Record::VertexSE2(Self::parse_vertex_se2(&parts, line_num)?),
            "VERTEX_SE3:QUAT" => Record::VertexSE3(Self::parse_vertex_se3(&parts, line_num)?),
            "EDGE_SE2" => Record::EdgeSE2(Self::parse_edge_se2(&parts, line_num)?),
            "EDGE_SE3:QUAT" => Record::EdgeSE3(Box::new(Self::parse_edge_se3(&parts, line_num)?)),
            other => Record::Unknown(other),
        };
        Ok(Some(record))
    }

    /// Parse VERTEX_SE2 line
    pub fn parse_vertex_se2(parts: &[&str], line_num: usize) -> GraphResult<VertexSE2> {
        const RECORD: &str = "VERTEX_SE2";
        require_fields(parts, 5, line_num, RECORD)?;
        let id = parse_field::<usize>(parts, 1, line_num, RECORD)?;
        let x = parse_field::<f64>(parts, 2, line_num, RECORD)?;
        let y = parse_field::<f64>(parts, 3, line_num, RECORD)?;
        let theta = parse_field::<f64>(parts, 4, line_num, RECORD)?;
        Ok(VertexSE2::new(id, x, y, theta))
    }

    /// Parse VERTEX_SE3:QUAT line
    pub fn parse_vertex_se3(parts: &[&str], line_num: usize) -> GraphResult<VertexSE3> {
        const RECORD: &str = "VERTEX_SE3:QUAT";
        require_fields(parts, 9, line_num, RECORD)?;
        let id = parse_field::<usize>(parts, 1, line_num, RECORD)?;
        let mut values = [0.0; 7];
        for (offset, value) in values.iter_mut().enumerate() {
            *value = parse_field::<f64>(parts, 2 + offset, line_num, RECORD)?;
        }
        let [x, y, z, qx, qy, qz, qw] = values;

        let translation = nalgebra::Vector3::new(x, y, z);
        let quaternion = nalgebra::Quaternion::new(qw, qx, qy, qz);

        let quat_norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
        if !quat_norm.is_finite() || (quat_norm - 1.0).abs() > 0.01 {
            return Err(GraphError::InvalidQuaternion {
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
    fn parse_edge_se2(parts: &[&str], line_num: usize) -> GraphResult<EdgeSE2> {
        const RECORD: &str = "EDGE_SE2";
        require_fields(parts, 12, line_num, RECORD)?;
        let from = parse_field::<usize>(parts, 1, line_num, RECORD)?;
        let to = parse_field::<usize>(parts, 2, line_num, RECORD)?;
        // Measurement (dx, dy, dtheta), then the upper triangle of the
        // information matrix: i11, i12, i13, i22, i23, i33.
        let mut values = [0.0; 9];
        for (offset, value) in values.iter_mut().enumerate() {
            *value = parse_field::<f64>(parts, 3 + offset, line_num, RECORD)?;
        }
        let [dx, dy, dtheta, i11, i12, i13, i22, i23, i33] = values;

        let information = nalgebra::Matrix3::new(i11, i12, i13, i12, i22, i23, i13, i23, i33);

        Ok(EdgeSE2::new(from, to, dx, dy, dtheta, information))
    }

    /// Parse EDGE_SE3:QUAT line: ids, translation, quaternion and the 21-value
    /// upper triangle of the information matrix.
    fn parse_edge_se3(parts: &[&str], line_num: usize) -> GraphResult<EdgeSE3> {
        const RECORD: &str = "EDGE_SE3:QUAT";
        require_fields(parts, 10, line_num, RECORD)?;
        let from = parse_field::<usize>(parts, 1, line_num, RECORD)?;
        let to = parse_field::<usize>(parts, 2, line_num, RECORD)?;
        let mut pose = [0.0; 7];
        for (offset, value) in pose.iter_mut().enumerate() {
            *value = parse_field::<f64>(parts, 3 + offset, line_num, RECORD)?;
        }
        let [tx, ty, tz, qx, qy, qz, qw] = pose;

        let translation = nalgebra::Vector3::new(tx, ty, tz);
        let rotation =
            nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));

        require_fields(parts, 31, line_num, RECORD)?;
        let mut upper = [0.0; 21];
        for (offset, value) in upper.iter_mut().enumerate() {
            *value = parse_field::<f64>(parts, 10 + offset, line_num, RECORD)?;
        }

        let mut information = nalgebra::Matrix6::zeros();
        let mut next = 0;
        for i in 0..6 {
            for j in i..6 {
                information[(i, j)] = upper[next];
                information[(j, i)] = upper[next];
                next += 1;
            }
        }

        Ok(EdgeSE3::new(from, to, translation, rotation, information))
    }
}

/// One parsed G2O line.
enum Record<'a> {
    VertexSE2(VertexSE2),
    VertexSE3(VertexSE3),
    EdgeSE2(EdgeSE2),
    EdgeSE3(Box<EdgeSE3>),
    /// A record type this loader does not read, e.g. `FIX` or `VERTEX_XY`.
    Unknown(&'a str),
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
        assert!(matches!(result, Err(GraphError::InvalidNumber { .. })));

        // Test missing fields
        let parts = vec!["VERTEX_SE2", "0"];
        let result = G2oLoader::parse_vertex_se2(&parts, 1);
        assert!(matches!(result, Err(GraphError::MissingFields { .. })));
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
            matches!(result, Err(GraphError::InvalidQuaternion { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::MissingFields { .. })),
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
            matches!(result, Err(GraphError::DuplicateVertex { id: 3 })),
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
            matches!(result, Err(GraphError::MissingFields { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::DuplicateVertex { id: 7 })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::MissingFields { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::MissingFields { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::InvalidNumber { .. })),
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
            matches!(result, Err(GraphError::DuplicateVertex { id: 0 })),
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

    #[test]
    fn test_edge_se3_with_a_short_information_matrix_is_an_error() -> TestResult {
        // 10 to 30 tokens used to pass the length check and then panic on the
        // information-matrix slice.
        for extra in [0usize, 1, 20] {
            let values = vec!["1.0"; extra].join(" ");
            let line = format!("EDGE_SE3:QUAT 0 1 0.0 0.0 0.0 0.0 0.0 0.0 1.0 {values}");
            let parts: Vec<&str> = line.split_whitespace().collect();
            let result = G2oLoader::parse_edge_se3(&parts, 7);
            assert!(
                matches!(
                    result,
                    Err(GraphError::MissingFields {
                        line: 7,
                        expected: 31,
                        ..
                    })
                ),
                "{} tokens must be refused, got {:?}",
                parts.len(),
                result.err()
            );
        }
        Ok(())
    }

    #[test]
    fn test_short_edge_se3_in_a_parallel_parse_is_an_error() -> TestResult {
        let mut f = NamedTempFile::new()?;
        for id in 0..MIN_LINES_FOR_PARALLEL + 10 {
            writeln!(f, "VERTEX_SE2 {id} 0.0 0.0 0.0")?;
        }
        writeln!(f, "EDGE_SE3:QUAT 0 1 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0 2.0")?;
        f.flush()?;
        let result = G2oLoader::load(f.path());
        assert!(
            matches!(result, Err(GraphError::MissingFields { .. })),
            "got {:?}",
            result.err()
        );
        Ok(())
    }

    /// Every entry of `actual` equals `expected` (the writer prints 17
    /// significant digits, so a round trip is exact).
    fn assert_same_matrix<const N: usize>(
        actual: &nalgebra::SMatrix<f64, N, N>,
        expected: &nalgebra::SMatrix<f64, N, N>,
    ) -> TestResult {
        for i in 0..N {
            for j in 0..N {
                if actual[(i, j)] != expected[(i, j)] {
                    return Err(format!(
                        "information ({i}, {j}) is {}, expected {}\n{actual}",
                        actual[(i, j)],
                        expected[(i, j)]
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn edge_se2_information_survives_a_round_trip() -> TestResult {
        let mut graph = Graph::new();
        graph
            .vertices_se2
            .insert(0, VertexSE2::new(0, 0.0, 0.0, 0.0));
        graph
            .vertices_se2
            .insert(1, VertexSE2::new(1, 1.0, 0.0, 0.0));
        // Every upper-triangle entry distinct, so any reordering shows.
        let info = Matrix3::new(11.0, 1.5, 0.25, 1.5, 22.0, -0.75, 0.25, -0.75, 33.0);
        graph
            .edges_se2
            .push(EdgeSE2::new(0, 1, 1.0, 0.5, 0.1, info));

        let f = NamedTempFile::new()?;
        G2oLoader::write(&graph, f.path())?;
        let loaded = G2oLoader::load(f.path())?;

        let edge = loaded
            .edges_se2
            .first()
            .ok_or("the edge was not read back")?;
        assert_same_matrix(&edge.information, &info)
    }

    #[test]
    fn edge_se3_information_survives_a_round_trip() -> TestResult {
        let rot = UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3);
        let mut graph = Graph::new();
        graph
            .vertices_se3
            .insert(0, VertexSE3::new(0, Vector3::zeros(), rot));
        graph
            .vertices_se3
            .insert(1, VertexSE3::new(1, Vector3::new(1.0, 2.0, 3.0), rot));
        // Symmetric, every upper-triangle entry distinct.
        let info = Matrix6::from_fn(|i, j| {
            let (low, high) = (i.min(j), i.max(j));
            if low == high {
                100.0 * (low + 1) as f64
            } else {
                0.5 * (low * 6 + high) as f64 - 4.0
            }
        });
        graph
            .edges_se3
            .push(EdgeSE3::new(0, 1, Vector3::new(1.0, 2.0, 3.0), rot, info));

        let f = NamedTempFile::new()?;
        G2oLoader::write(&graph, f.path())?;
        let loaded = G2oLoader::load(f.path())?;

        let edge = loaded
            .edges_se3
            .first()
            .ok_or("the edge was not read back")?;
        assert_same_matrix(&edge.information, &info)
    }

    /// The reader follows g2o's `EdgeSE2::read`: the upper triangle row by
    /// row, `I11 I12 I13 I22 I23 I33`. The line is the first edge of the
    /// Intel dataset, whose only off-diagonal entry is `I12`.
    #[test]
    fn edge_se2_information_is_read_in_g2o_order() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(
            f,
            "EDGE_SE2 0 1 0.000000 0.000000 -0.000642 \
             11.111271 -0.249667 0.000000 399.999840 0.000000 2496.793089"
        )?;
        f.flush()?;
        let graph = G2oLoader::load(f.path())?;
        let edge = graph.edges_se2.first().ok_or("the edge was not read")?;
        let expected = Matrix3::new(
            11.111271,
            -0.249667,
            0.0,
            -0.249667,
            399.999840,
            0.0,
            0.0,
            0.0,
            2496.793089,
        );
        assert_same_matrix(&edge.information, &expected)
    }
}
