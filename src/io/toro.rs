use crate::io::{ApexSolverIoError, EdgeSE2, Graph, GraphLoader, VertexSE2};
use faer::mat;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// TORO format loader
pub struct ToroLoader;

impl GraphLoader for ToroLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<Graph, ApexSolverIoError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let content = std::str::from_utf8(&mmap).map_err(|e| ApexSolverIoError::Parse {
            line: 0,
            message: format!("Invalid UTF-8: {e}"),
        })?;

        Self::parse_content(content)
    }

    fn write<P: AsRef<Path>>(_graph: &Graph, _path: P) -> Result<(), ApexSolverIoError> {
        // TODO: Implement TORO writing
        Err(ApexSolverIoError::UnsupportedFormat(
            "TORO writing not implemented yet".to_string(),
        ))
    }
}

impl ToroLoader {
    fn parse_content(content: &str) -> Result<Graph, ApexSolverIoError> {
        let lines: Vec<&str> = content.lines().collect();
        let mut graph = Graph::new();

        for (line_num, line) in lines.iter().enumerate() {
            Self::parse_line(line, line_num + 1, &mut graph)?;
        }

        Ok(graph)
    }

    fn parse_line(line: &str, line_num: usize, graph: &mut Graph) -> Result<(), ApexSolverIoError> {
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
                    return Err(ApexSolverIoError::DuplicateVertex { id });
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

    fn parse_vertex2(parts: &[&str], line_num: usize) -> Result<VertexSE2, ApexSolverIoError> {
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

    fn parse_edge2(parts: &[&str], line_num: usize) -> Result<EdgeSE2, ApexSolverIoError> {
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

        // Parse TORO information matrix (I11, I12, I22, I33, I13, I23)
        let i11 = parts[6]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[6].to_string(),
            })?;
        let i12 = parts[7]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[7].to_string(),
            })?;
        let i22 = parts[8]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[8].to_string(),
            })?;
        let i33 = parts[9]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[9].to_string(),
            })?;
        let i13 = parts[10]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[10].to_string(),
            })?;
        let i23 = parts[11]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[11].to_string(),
            })?;

        let information = mat![[i11, i12, i13], [i12, i22, i23], [i13, i23, i33]];

        Ok(EdgeSE2::new(from, to, dx, dy, dtheta, information))
    }
}
