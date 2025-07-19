use nalgebra::{Matrix3, Matrix6, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

// Import manifold types
use crate::manifold::se2::SE2;
use crate::manifold::se3::SE3;

// Module declarations
pub mod g2o;
pub mod toro;
pub mod tum;

// Re-exports
pub use g2o::G2oLoader;
pub use toro::ToroLoader;
pub use tum::TumLoader;

/// Errors that can occur during graph file parsing
#[derive(Error, Debug)]
pub enum ApexSolverIoError {
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

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),
}

/// SE2 vertex with ID (x, y, theta)
#[derive(Debug, Clone, PartialEq)]
pub struct VertexSE2 {
    pub id: usize,
    pub pose: SE2,
}

impl VertexSE2 {
    pub fn new(id: usize, x: f64, y: f64, theta: f64) -> Self {
        Self {
            id,
            pose: SE2::from_xy_angle(x, y, theta),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn x(&self) -> f64 {
        self.pose.x()
    }

    pub fn y(&self) -> f64 {
        self.pose.y()
    }

    pub fn theta(&self) -> f64 {
        self.pose.angle()
    }
}

/// SE3 vertex with ID (x, y, z, qx, qy, qz, qw)
#[derive(Debug, Clone, PartialEq)]
pub struct VertexSE3 {
    pub id: usize,
    pub pose: SE3,
}

impl VertexSE3 {
    pub fn new(id: usize, translation: Vector3<f64>, rotation: UnitQuaternion<f64>) -> Self {
        Self {
            id,
            pose: SE3::new(translation, rotation),
        }
    }

    pub fn from_translation_quaternion(
        id: usize,
        x: f64,
        y: f64,
        z: f64,
        qw: f64,
        qx: f64,
        qy: f64,
        qz: f64,
    ) -> Self {
        Self {
            id,
            pose: SE3::from_translation_quaternion(x, y, z, qw, qx, qy, qz),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn translation(&self) -> Vector3<f64> {
        self.pose.translation()
    }

    pub fn rotation(&self) -> UnitQuaternion<f64> {
        self.pose.rotation_quaternion()
    }

    pub fn x(&self) -> f64 {
        self.pose.x()
    }

    pub fn y(&self) -> f64 {
        self.pose.y()
    }

    pub fn z(&self) -> f64 {
        self.pose.z()
    }
}

/// TUM trajectory vertex with timestamp
#[derive(Debug, Clone, PartialEq)]
pub struct VertexTUM {
    pub timestamp: f64,
    pub pose: SE3,
}

impl VertexTUM {
    pub fn new(timestamp: f64, translation: Vector3<f64>, rotation: UnitQuaternion<f64>) -> Self {
        Self {
            timestamp,
            pose: SE3::new(translation, rotation),
        }
    }

    pub fn from_translation_quaternion(
        timestamp: f64,
        x: f64,
        y: f64,
        z: f64,
        qw: f64,
        qx: f64,
        qy: f64,
        qz: f64,
    ) -> Self {
        Self {
            timestamp,
            pose: SE3::from_translation_quaternion(x, y, z, qw, qx, qy, qz),
        }
    }

    pub fn translation(&self) -> Vector3<f64> {
        self.pose.translation()
    }

    pub fn rotation(&self) -> UnitQuaternion<f64> {
        self.pose.rotation_quaternion()
    }
}

/// 2D edge constraint between two SE2 vertices
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSE2 {
    pub from: usize,
    pub to: usize,
    pub measurement: SE2,          // Relative transformation
    pub information: Matrix3<f64>, // 3x3 information matrix
}

impl EdgeSE2 {
    pub fn new(
        from: usize,
        to: usize,
        dx: f64,
        dy: f64,
        dtheta: f64,
        information: Matrix3<f64>,
    ) -> Self {
        Self {
            from,
            to,
            measurement: SE2::from_xy_angle(dx, dy, dtheta),
            information,
        }
    }
}

/// 3D edge constraint between two SE3 vertices
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSE3 {
    pub from: usize,
    pub to: usize,
    pub measurement: SE3,          // Relative transformation
    pub information: Matrix6<f64>, // 6x6 information matrix
}

impl EdgeSE3 {
    pub fn new(
        from: usize,
        to: usize,
        translation: Vector3<f64>,
        rotation: UnitQuaternion<f64>,
        information: Matrix6<f64>,
    ) -> Self {
        Self {
            from,
            to,
            measurement: SE3::new(translation, rotation),
            information,
        }
    }
}

/// Main graph structure containing vertices and edges
#[derive(Debug, Clone)]
pub struct G2oGraph {
    pub vertices_se2: HashMap<usize, VertexSE2>,
    pub vertices_se3: HashMap<usize, VertexSE3>,
    pub vertices_tum: Vec<VertexTUM>,
    pub edges_se2: Vec<EdgeSE2>,
    pub edges_se3: Vec<EdgeSE3>,
}

impl G2oGraph {
    pub fn new() -> Self {
        Self {
            vertices_se2: HashMap::new(),
            vertices_se3: HashMap::new(),
            vertices_tum: Vec::new(),
            edges_se2: Vec::new(),
            edges_se3: Vec::new(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices_se2.len() + self.vertices_se3.len() + self.vertices_tum.len()
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

/// Trait for graph file loaders and writers
pub trait GraphLoader {
    /// Load a graph from a file
    fn load<P: AsRef<Path>>(path: P) -> Result<G2oGraph, ApexSolverIoError>;

    /// Write a graph to a file
    fn write<P: AsRef<Path>>(graph: &G2oGraph, path: P) -> Result<(), ApexSolverIoError>;
}

/// Convenience function to load any supported format based on file extension
pub fn load_graph<P: AsRef<Path>>(path: P) -> Result<G2oGraph, ApexSolverIoError> {
    let path_ref = path.as_ref();
    let extension = path_ref
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| ApexSolverIoError::UnsupportedFormat("No file extension".to_string()))?;

    match extension.to_lowercase().as_str() {
        "g2o" => G2oLoader::load(path),
        "graph" => ToroLoader::load(path),
        "txt" | "csv" => TumLoader::load(path),
        _ => Err(ApexSolverIoError::UnsupportedFormat(format!(
            "Unsupported extension: {extension}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_simple_graph() -> Result<(), ApexSolverIoError> {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "VERTEX_SE2 0 0.0 0.0 0.0")?;
        writeln!(temp_file, "VERTEX_SE2 1 1.0 0.0 0.0")?;
        writeln!(temp_file, "# This is a comment")?;
        writeln!(temp_file)?; // Empty line
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
        println!(
            "M3500 loaded: {} vertices, {} edges",
            graph.vertex_count(),
            graph.edge_count()
        );
        assert!(!graph.vertices_se2.is_empty());
        Ok(())
    }

    #[test]
    fn test_load_parking_garage() -> Result<(), Box<dyn std::error::Error>> {
        let graph = G2oLoader::load("data/parking-garage.g2o")?;
        println!(
            "Parking garage loaded: {} vertices, {} edges",
            graph.vertex_count(),
            graph.edge_count()
        );
        assert!(!graph.vertices_se3.is_empty());
        Ok(())
    }

    #[test]
    fn test_load_sphere2500() -> Result<(), Box<dyn std::error::Error>> {
        let graph = G2oLoader::load("data/sphere2500.g2o")?;
        println!(
            "Sphere2500 loaded: {} vertices, {} edges",
            graph.vertex_count(),
            graph.edge_count()
        );
        assert!(!graph.vertices_se3.is_empty());
        Ok(())
    }

    #[test]
    fn test_duplicate_vertex_error() -> Result<(), std::io::Error> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "VERTEX_SE2 0 0.0 0.0 0.0")?;
        writeln!(temp_file, "VERTEX_SE2 0 1.0 0.0 0.0")?; // Duplicate ID

        let result = G2oLoader::load(temp_file.path());
        assert!(matches!(
            result,
            Err(ApexSolverIoError::DuplicateVertex { id: 0 })
        ));

        Ok(())
    }

    #[test]
    fn test_toro_loader() -> Result<(), std::io::Error> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "VERTEX2 0 0.0 0.0 0.0")?;
        writeln!(temp_file, "VERTEX2 1 1.0 0.0 0.0")?;

        let graph = ToroLoader::load(temp_file.path()).unwrap();
        assert_eq!(graph.vertices_se2.len(), 2);

        Ok(())
    }

    #[test]
    fn test_tum_loader() -> Result<(), std::io::Error> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0")?;
        writeln!(temp_file, "2.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0")?;

        let graph = TumLoader::load(temp_file.path()).unwrap();
        assert_eq!(graph.vertices_tum.len(), 2);

        Ok(())
    }
}
