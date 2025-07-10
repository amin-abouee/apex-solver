use super::*;
use memmap2::Mmap;
use std::fs::File;

/// TUM format loader
pub struct TumLoader;

impl GraphLoader for TumLoader {
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
        // TODO: Implement TUM writing
        Err(ApexSolverIoError::UnsupportedFormat(
            "TUM writing not implemented yet".to_string(),
        ))
    }
}

impl TumLoader {
    fn parse_content(content: &str) -> Result<G2oGraph, ApexSolverIoError> {
        let lines: Vec<&str> = content.lines().collect();
        let mut graph = G2oGraph::new();

        for (line_num, line) in lines.iter().enumerate() {
            Self::parse_line(line, line_num + 1, &mut graph)?;
        }

        Ok(graph)
    }

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
        if parts.len() < 8 {
            return Err(ApexSolverIoError::MissingFields { line: line_num });
        }

        let vertex = Self::parse_tum_vertex(&parts, line_num)?;
        graph.vertices_tum.push(vertex);

        Ok(())
    }

    fn parse_tum_vertex(parts: &[&str], line_num: usize) -> Result<VertexTUM, ApexSolverIoError> {
        // TUM format: timestamp x y z q_x q_y q_z q_w
        let timestamp = parts[0]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[0].to_string(),
            })?;

        let x = parts[1]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[1].to_string(),
            })?;

        let y = parts[2]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[2].to_string(),
            })?;

        let z = parts[3]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[3].to_string(),
            })?;

        let qx = parts[4]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[4].to_string(),
            })?;

        let qy = parts[5]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[5].to_string(),
            })?;

        let qz = parts[6]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[6].to_string(),
            })?;

        let qw = parts[7]
            .parse::<f64>()
            .map_err(|_| ApexSolverIoError::InvalidNumber {
                line: line_num,
                value: parts[7].to_string(),
            })?;

        let translation = Vector3::new(x, y, z);
        let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));

        Ok(VertexTUM {
            timestamp,
            translation,
            rotation,
        })
    }
}
