use super::*;
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;

/// High-performance G2O file loader
pub struct G2oLoader;

impl GraphLoader for G2oLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<G2oGraph, ApexSolverIoError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let content = std::str::from_utf8(&mmap).map_err(|e| ApexSolverIoError::Parse {
            line: 0,
            message: format!("Invalid UTF-8: {e}"),
        })?;

        Self::parse_content(content)
    }

    fn write<P: AsRef<Path>>(_graph: &G2oGraph, _path: P) -> Result<(), ApexSolverIoError> {
        // TODO: Implement G2O writing
        Err(ApexSolverIoError::UnsupportedFormat(
            "G2O writing not implemented yet".to_string(),
        ))
    }
}

impl G2oLoader {
    /// Parse G2O content with performance optimizations
    fn parse_content(content: &str) -> Result<G2oGraph, ApexSolverIoError> {
        let lines: Vec<&str> = content.lines().collect();

        // Pre-allocate collections based on estimated size
        let estimated_vertices = lines.len() / 2; // Rough estimate
        let mut graph = G2oGraph {
            vertices_se2: HashMap::with_capacity(estimated_vertices),
            vertices_se3: HashMap::with_capacity(estimated_vertices),
            vertices_tum: Vec::new(),
            edges_se2: Vec::with_capacity(estimated_vertices / 4),
            edges_se3: Vec::with_capacity(estimated_vertices / 4),
        };

        // For large files, use parallel processing
        if lines.len() > 5000 {
            Self::parse_parallel(&lines, &mut graph)?;
        } else {
            Self::parse_sequential(&lines, &mut graph)?;
        }

        Ok(graph)
    }

    /// Sequential parsing for smaller files
    fn parse_sequential(lines: &[&str], graph: &mut G2oGraph) -> Result<(), ApexSolverIoError> {
        for (line_num, line) in lines.iter().enumerate() {
            Self::parse_line(line, line_num + 1, graph)?;
        }
        Ok(())
    }

    /// Parallel parsing for larger files
    fn parse_parallel(lines: &[&str], graph: &mut G2oGraph) -> Result<(), ApexSolverIoError> {
        // Collect parse results in parallel
        let results: Result<Vec<_>, ApexSolverIoError> = lines
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
                        return Err(ApexSolverIoError::DuplicateVertex { id });
                    }
                }
                ParsedItem::VertexSE3(vertex) => {
                    let id = vertex.id;
                    if graph.vertices_se3.insert(id, vertex).is_some() {
                        return Err(ApexSolverIoError::DuplicateVertex { id });
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
    fn parse_line(
        line: &str,
        line_num: usize,
        graph: &mut G2oGraph,
    ) -> Result<(), ApexSolverIoError> {
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
                    return Err(ApexSolverIoError::DuplicateVertex { id });
                }
            }
            "VERTEX_SE3:QUAT" => {
                let vertex = Self::parse_vertex_se3(&parts, line_num)?;
                let id = vertex.id;
                if graph.vertices_se3.insert(id, vertex).is_some() {
                    return Err(ApexSolverIoError::DuplicateVertex { id });
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
    fn parse_line_to_enum(
        line: &str,
        line_num: usize,
    ) -> Result<Option<ParsedItem>, ApexSolverIoError> {
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
    pub fn parse_vertex_se2(
        parts: &[&str],
        line_num: usize,
    ) -> Result<VertexSE2, ApexSolverIoError> {
        if parts.len() < 5 {
            return Err(ApexSolverIoError::MissingFields { line: line_num });
        }

        let id = parts[1]
            .parse::<usize>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[1].to_string(),
            })?;

        let x = parts[2]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[2].to_string(),
            })?;

        let y = parts[3]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[3].to_string(),
            })?;

        let theta = parts[4]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[4].to_string(),
            })?;

        Ok(VertexSE2::new(id, x, y, theta))
    }

    /// Parse VERTEX_SE3:QUAT line
    pub fn parse_vertex_se3(
        parts: &[&str],
        line_num: usize,
    ) -> Result<VertexSE3, ApexSolverIoError> {
        if parts.len() < 9 {
            return Err(ApexSolverIoError::MissingFields { line: line_num });
        }

        let id = parts[1]
            .parse::<usize>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[1].to_string(),
            })?;

        let x = parts[2]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[2].to_string(),
            })?;

        let y = parts[3]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[3].to_string(),
            })?;

        let z = parts[4]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[4].to_string(),
            })?;

        let qx = parts[5]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[5].to_string(),
            })?;

        let qy = parts[6]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[6].to_string(),
            })?;

        let qz = parts[7]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[7].to_string(),
            })?;

        let qw = parts[8]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[8].to_string(),
            })?;

        let translation = Vector3::new(x, y, z);
        let quaternion = Quaternion::new(qw, qx, qy, qz);
        Ok(VertexSE3::from_translation_quaternion(
            id,
            translation,
            quaternion,
        ))
    }

    /// Parse EDGE_SE2 line
    fn parse_edge_se2(parts: &[&str], line_num: usize) -> Result<EdgeSE2, ApexSolverIoError> {
        if parts.len() < 12 {
            return Err(ApexSolverIoError::MissingFields { line: line_num });
        }

        let from = parts[1]
            .parse::<usize>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[1].to_string(),
            })?;

        let to = parts[2]
            .parse::<usize>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[2].to_string(),
            })?;

        // Parse measurement (dx, dy, dtheta)
        let dx = parts[3]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[3].to_string(),
            })?;
        let dy = parts[4]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[4].to_string(),
            })?;
        let dtheta = parts[5]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[5].to_string(),
            })?;

        // Parse information matrix (upper triangular: i11, i12, i13, i22, i23, i33)
        let info_values: Result<Vec<f64>, _> =
            parts[6..12].iter().map(|s| s.parse::<f64>()).collect();

        let info_values = info_values.map_err(|_| ApexSolverIoError::Parse {
            line: line_num,
            message: "Invalid information matrix values".to_string(),
        })?;

        let information = Matrix3::new(
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
    fn parse_edge_se3(parts: &[&str], line_num: usize) -> Result<EdgeSE3, ApexSolverIoError> {
        // EDGE_SE3:QUAT from_id to_id tx ty tz qx qy qz qw [information matrix values]
        if parts.len() < 10 {
            return Err(ApexSolverIoError::MissingFields { line: line_num });
        }

        // Parse vertex IDs
        let from = parts[1]
            .parse::<usize>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[1].to_string(),
            })?;

        let to = parts[2]
            .parse::<usize>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[2].to_string(),
            })?;

        // Parse translation (tx, ty, tz)
        let tx = parts[3]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[3].to_string(),
            })?;

        let ty = parts[4]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[4].to_string(),
            })?;

        let tz = parts[5]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[5].to_string(),
            })?;

        let translation = Vector3::new(tx, ty, tz);

        // Parse rotation quaternion (qx, qy, qz, qw)
        let qx = parts[6]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[6].to_string(),
            })?;

        let qy = parts[7]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[7].to_string(),
            })?;

        let qz = parts[8]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[8].to_string(),
            })?;

        let qw = parts[9]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[9].to_string(),
            })?;

        let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));

        // For now, use identity information matrix (could parse full 6x6 matrix if needed)
        let information = Matrix6::identity();

        Ok(EdgeSE3::new(from, to, translation, rotation, information))
    }
}

/// Enum for parsed items (used in parallel processing)
#[derive(Debug)]
enum ParsedItem {
    VertexSE2(VertexSE2),
    VertexSE3(VertexSE3),
    EdgeSE2(EdgeSE2),
    EdgeSE3(Box<EdgeSE3>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vertex_se2() {
        let parts = vec!["VERTEX_SE2", "0", "1.0", "2.0", "0.5"];
        let vertex = G2oLoader::parse_vertex_se2(&parts, 1).unwrap();

        assert_eq!(vertex.id(), 0);
        assert_eq!(vertex.x(), 1.0);
        assert_eq!(vertex.y(), 2.0);
        assert_eq!(vertex.theta(), 0.5);
    }

    #[test]
    fn test_parse_vertex_se3() {
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
        let vertex = G2oLoader::parse_vertex_se3(&parts, 1).unwrap();

        assert_eq!(vertex.id(), 1);
        assert_eq!(vertex.translation(), Vector3::new(1.0, 2.0, 3.0));
        assert!(vertex.rotation().quaternion().w > 0.99); // Should be identity quaternion
    }

    #[test]
    fn test_error_handling() {
        // Test invalid number
        let parts = vec!["VERTEX_SE2", "invalid", "1.0", "2.0", "0.5"];
        let result = G2oLoader::parse_vertex_se2(&parts, 1);
        assert!(matches!(
            result,
            Err(ApexSolverIoError::InvalidNumber { .. })
        ));

        // Test missing fields
        let parts = vec!["VERTEX_SE2", "0"];
        let result = G2oLoader::parse_vertex_se2(&parts, 1);
        assert!(matches!(
            result,
            Err(ApexSolverIoError::MissingFields { .. })
        ));
    }
}
