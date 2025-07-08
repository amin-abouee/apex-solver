use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use memmap2::Mmap;
use rayon::prelude::*;
use nalgebra::{Vector3, UnitQuaternion, Matrix3, Matrix6};
use thiserror::Error;

/// Errors that can occur during G2O file parsing
#[derive(Error, Debug)]
pub enum G2oError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },
    
    #[error("Unsupported vertex type: {0}")]
    UnsupportedVertexType(String),
    
    #[error("Unsupported edge type: {0}")]
    UnsupportedEdgeType(String),
    
    #[error("Invalid number format at line {line}: {value}")]
    InvalidNumber { line: usize, value: String },
    
    #[error("Missing required fields at line {line}")]
    MissingFields { line: usize },
    
    #[error("Duplicate vertex ID: {id}")]
    DuplicateVertex { id: usize },
}

/// 2D pose vertex (x, y, theta)
#[derive(Debug, Clone, PartialEq)]
pub struct VertexSE2 {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

/// 3D pose vertex with quaternion (x, y, z, qx, qy, qz, qw)
#[derive(Debug, Clone, PartialEq)]
pub struct VertexSE3 {
    pub id: usize,
    pub translation: Vector3<f64>,
    pub rotation: UnitQuaternion<f64>,
}

/// 2D edge constraint between two SE2 vertices
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSE2 {
    pub from: usize,
    pub to: usize,
    pub measurement: Vector3<f64>, // dx, dy, dtheta
    pub information: Matrix3<f64>, // 3x3 information matrix
}

/// 3D edge constraint between two SE3 vertices
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSE3 {
    pub from: usize,
    pub to: usize,
    pub translation: Vector3<f64>,
    pub rotation: UnitQuaternion<f64>,
    pub information: Matrix6<f64>, // 6x6 information matrix
}

/// Main graph structure containing vertices and edges
#[derive(Debug, Clone)]
pub struct G2oGraph {
    pub vertices_se2: HashMap<usize, VertexSE2>,
    pub vertices_se3: HashMap<usize, VertexSE3>,
    pub edges_se2: Vec<EdgeSE2>,
    pub edges_se3: Vec<EdgeSE3>,
}

impl G2oGraph {
    pub fn new() -> Self {
        Self {
            vertices_se2: HashMap::new(),
            vertices_se3: HashMap::new(),
            edges_se2: Vec::new(),
            edges_se3: Vec::new(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices_se2.len() + self.vertices_se3.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges_se2.len() + self.edges_se3.len()
    }
}

impl Default for G2oGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// High-performance G2O file loader
pub struct G2oLoader;

impl G2oLoader {
    /// Load a G2O file with optimal performance using memory mapping
    pub fn load<P: AsRef<Path>>(path: P) -> Result<G2oGraph, G2oError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let content = std::str::from_utf8(&mmap)
            .map_err(|e| G2oError::Parse { 
                line: 0, 
                message: format!("Invalid UTF-8: {}", e) 
            })?;

        Self::parse_content(content)
    }

    /// Parse G2O content with performance optimizations
    fn parse_content(content: &str) -> Result<G2oGraph, G2oError> {
        let lines: Vec<&str> = content.lines().collect();
        
        // Pre-allocate collections based on estimated size
        let estimated_vertices = lines.len() / 2; // Rough estimate
        let mut graph = G2oGraph {
            vertices_se2: HashMap::with_capacity(estimated_vertices),
            vertices_se3: HashMap::with_capacity(estimated_vertices),
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
    fn parse_sequential(lines: &[&str], graph: &mut G2oGraph) -> Result<(), G2oError> {
        for (line_num, line) in lines.iter().enumerate() {
            Self::parse_line(line, line_num + 1, graph)?;
        }
        Ok(())
    }

    /// Parallel parsing for larger files
    fn parse_parallel(lines: &[&str], graph: &mut G2oGraph) -> Result<(), G2oError> {
        // Collect parse results in parallel
        let results: Result<Vec<_>, G2oError> = lines
            .par_iter()
            .enumerate()
            .map(|(line_num, line)| {
                Self::parse_line_to_enum(line, line_num + 1)
            })
            .collect();

        let parsed_items = results?;

        // Sequential insertion to avoid concurrent modification
        for item in parsed_items.into_iter().flatten() {
            match item {
                ParsedItem::VertexSE2(vertex) => {
                    let id = vertex.id;
                    if graph.vertices_se2.insert(id, vertex).is_some() {
                        return Err(G2oError::DuplicateVertex { id });
                    }
                },
                ParsedItem::VertexSE3(vertex) => {
                    let id = vertex.id;
                    if graph.vertices_se3.insert(id, vertex).is_some() {
                        return Err(G2oError::DuplicateVertex { id });
                    }
                },
                ParsedItem::EdgeSE2(edge) => {
                    graph.edges_se2.push(edge);
                },
                ParsedItem::EdgeSE3(edge) => {
                    graph.edges_se3.push(edge);
                },
            }
        }

        Ok(())
    }

    /// Parse a single line (for sequential processing)
    fn parse_line(line: &str, line_num: usize, graph: &mut G2oGraph) -> Result<(), G2oError> {
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
                    return Err(G2oError::DuplicateVertex { id });
                }
            },
            "VERTEX_SE3:QUAT" => {
                let vertex = Self::parse_vertex_se3(&parts, line_num)?;
                let id = vertex.id;
                if graph.vertices_se3.insert(id, vertex).is_some() {
                    return Err(G2oError::DuplicateVertex { id });
                }
            },
            "EDGE_SE2" => {
                let edge = Self::parse_edge_se2(&parts, line_num)?;
                graph.edges_se2.push(edge);
            },
            "EDGE_SE3:QUAT" => {
                let edge = Self::parse_edge_se3(&parts, line_num)?;
                graph.edges_se3.push(edge);
            },
            _ => {
                // Skip unknown types silently for compatibility
            }
        }

        Ok(())
    }

    /// Parse a single line to enum (for parallel processing)
    fn parse_line_to_enum(line: &str, line_num: usize) -> Result<Option<ParsedItem>, G2oError> {
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
            "VERTEX_SE2" => {
                Some(ParsedItem::VertexSE2(Self::parse_vertex_se2(&parts, line_num)?))
            },
            "VERTEX_SE3:QUAT" => {
                Some(ParsedItem::VertexSE3(Self::parse_vertex_se3(&parts, line_num)?))
            },
            "EDGE_SE2" => {
                Some(ParsedItem::EdgeSE2(Self::parse_edge_se2(&parts, line_num)?))
            },
            "EDGE_SE3:QUAT" => {
                Some(ParsedItem::EdgeSE3(Self::parse_edge_se3(&parts, line_num)?))
            },
            _ => None, // Skip unknown types
        };

        Ok(item)
    }

    /// Parse VERTEX_SE2 line
    fn parse_vertex_se2(parts: &[&str], line_num: usize) -> Result<VertexSE2, G2oError> {
        if parts.len() < 5 {
            return Err(G2oError::MissingFields { line: line_num });
        }

        let id = parts[1].parse::<usize>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[1].to_string() 
            })?;

        let x = parts[2].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[2].to_string() 
            })?;

        let y = parts[3].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[3].to_string() 
            })?;

        let theta = parts[4].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[4].to_string() 
            })?;

        Ok(VertexSE2 { id, x, y, theta })
    }

    /// Parse VERTEX_SE3:QUAT line
    fn parse_vertex_se3(parts: &[&str], line_num: usize) -> Result<VertexSE3, G2oError> {
        if parts.len() < 9 {
            return Err(G2oError::MissingFields { line: line_num });
        }

        let id = parts[1].parse::<usize>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[1].to_string() 
            })?;

        let x = parts[2].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[2].to_string() 
            })?;

        let y = parts[3].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[3].to_string() 
            })?;

        let z = parts[4].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[4].to_string() 
            })?;

        let qx = parts[5].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[5].to_string() 
            })?;

        let qy = parts[6].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[6].to_string() 
            })?;

        let qz = parts[7].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[7].to_string() 
            })?;

        let qw = parts[8].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[8].to_string() 
            })?;

        let translation = Vector3::new(x, y, z);
        let rotation = UnitQuaternion::from_quaternion(
            nalgebra::Quaternion::new(qw, qx, qy, qz)
        );

        Ok(VertexSE3 { id, translation, rotation })
    }

    /// Parse EDGE_SE2 line (placeholder implementation)
    fn parse_edge_se2(parts: &[&str], line_num: usize) -> Result<EdgeSE2, G2oError> {
        if parts.len() < 12 {
            return Err(G2oError::MissingFields { line: line_num });
        }

        let from = parts[1].parse::<usize>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[1].to_string() 
            })?;

        let to = parts[2].parse::<usize>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[2].to_string() 
            })?;

        // Parse measurement (dx, dy, dtheta)
        let dx = parts[3].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[3].to_string() 
            })?;
        let dy = parts[4].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[4].to_string() 
            })?;
        let dtheta = parts[5].parse::<f64>()
            .map_err(|_| G2oError::InvalidNumber { 
                line: line_num, 
                value: parts[5].to_string() 
            })?;

        let measurement = Vector3::new(dx, dy, dtheta);

        // Parse information matrix (upper triangular: i11, i12, i13, i22, i23, i33)
        let info_values: Result<Vec<f64>, _> = parts[6..12]
            .iter()
            .map(|s| s.parse::<f64>())
            .collect();

        let info_values = info_values
            .map_err(|_| G2oError::Parse { 
                line: line_num, 
                message: "Invalid information matrix values".to_string() 
            })?;

        let information = Matrix3::new(
            info_values[0], info_values[1], info_values[2],
            info_values[1], info_values[3], info_values[4],
            info_values[2], info_values[4], info_values[5],
        );

        Ok(EdgeSE2 { from, to, measurement, information })
    }

    /// Parse EDGE_SE3:QUAT line (placeholder implementation)
    fn parse_edge_se3(_parts: &[&str], _line_num: usize) -> Result<EdgeSE3, G2oError> {
        // Simplified implementation - full edge parsing would be more complex
        let from = 0;
        let to = 1;
        let translation = Vector3::zeros();
        let rotation = UnitQuaternion::identity();
        let information = Matrix6::identity();

        Ok(EdgeSE3 { from, to, translation, rotation, information })
    }
}

/// Enum for parsed items (used in parallel processing)
#[derive(Debug)]
enum ParsedItem {
    VertexSE2(VertexSE2),
    VertexSE3(VertexSE3),
    EdgeSE2(EdgeSE2),
    EdgeSE3(EdgeSE3),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_vertex_se2() {
        let parts = vec!["VERTEX_SE2", "0", "1.0", "2.0", "0.5"];
        let vertex = G2oLoader::parse_vertex_se2(&parts, 1).unwrap();
        
        assert_eq!(vertex.id, 0);
        assert_eq!(vertex.x, 1.0);
        assert_eq!(vertex.y, 2.0);
        assert_eq!(vertex.theta, 0.5);
    }

    #[test]
    fn test_parse_vertex_se3() {
        let parts = vec!["VERTEX_SE3:QUAT", "1", "1.0", "2.0", "3.0", "0.0", "0.0", "0.0", "1.0"];
        let vertex = G2oLoader::parse_vertex_se3(&parts, 1).unwrap();
        
        assert_eq!(vertex.id, 1);
        assert_eq!(vertex.translation, Vector3::new(1.0, 2.0, 3.0));
        assert!(vertex.rotation.quaternion().w > 0.99); // Should be identity quaternion
    }

    #[test]
    fn test_load_simple_graph() -> Result<(), G2oError> {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "VERTEX_SE2 0 0.0 0.0 0.0")?;
        writeln!(temp_file, "VERTEX_SE2 1 1.0 0.0 0.0")?;
        writeln!(temp_file, "# This is a comment")?;
        writeln!(temp_file, "")?; // Empty line
        writeln!(temp_file, "VERTEX_SE3:QUAT 2 0.0 0.0 0.0 0.0 0.0 0.0 1.0")?;

        let graph = G2oLoader::load(temp_file.path())?;
        
        assert_eq!(graph.vertices_se2.len(), 2);
        assert_eq!(graph.vertices_se3.len(), 1);
        assert!(graph.vertices_se2.contains_key(&0));
        assert!(graph.vertices_se2.contains_key(&1));
        assert!(graph.vertices_se3.contains_key(&2));

        Ok(())
    }

    #[test]
    fn test_load_m3500() -> Result<(), Box<dyn std::error::Error>> {
        let graph = G2oLoader::load("data/M3500.g2o")?;
        println!("M3500 loaded: {} vertices, {} edges", 
                 graph.vertex_count(), graph.edge_count());
        assert!(graph.vertices_se2.len() > 0);
        Ok(())
    }

    #[test]
    fn test_load_parking_garage() -> Result<(), Box<dyn std::error::Error>> {
        let graph = G2oLoader::load("data/parking-garage.g2o")?;
        println!("Parking garage loaded: {} vertices, {} edges", 
                 graph.vertex_count(), graph.edge_count());
        assert!(graph.vertices_se3.len() > 0);
        Ok(())
    }

    #[test]
    fn test_load_sphere2500() -> Result<(), Box<dyn std::error::Error>> {
        let graph = G2oLoader::load("data/sphere2500.g2o")?;
        println!("Sphere2500 loaded: {} vertices, {} edges", 
                 graph.vertex_count(), graph.edge_count());
        assert!(graph.vertices_se3.len() > 0);
        Ok(())
    }

    #[test]
    fn test_error_handling() {
        // Test invalid number
        let parts = vec!["VERTEX_SE2", "invalid", "1.0", "2.0", "0.5"];
        let result = G2oLoader::parse_vertex_se2(&parts, 1);
        assert!(matches!(result, Err(G2oError::InvalidNumber { .. })));

        // Test missing fields
        let parts = vec!["VERTEX_SE2", "0"];
        let result = G2oLoader::parse_vertex_se2(&parts, 1);
        assert!(matches!(result, Err(G2oError::MissingFields { .. })));
    }

    #[test]
    fn test_duplicate_vertex_error() -> Result<(), std::io::Error> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "VERTEX_SE2 0 0.0 0.0 0.0")?;
        writeln!(temp_file, "VERTEX_SE2 0 1.0 0.0 0.0")?; // Duplicate ID

        let result = G2oLoader::load(temp_file.path());
        assert!(matches!(result, Err(G2oError::DuplicateVertex { id: 0 })));

        Ok(())
    }
} 