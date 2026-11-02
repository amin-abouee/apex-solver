//! Errors raised while reading or writing a pose-graph file (G2O or TORO).

use std::io;
use std::path::PathBuf;

/// Result of a pose-graph read or write.
pub type GraphResult<T> = Result<T, GraphError>;

/// A pose-graph file that cannot be read, parsed or written.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GraphError {
    #[error("failed to read graph file {}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to write graph file {}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("line {line}: {record} has {actual} fields, expected at least {expected}")]
    MissingFields {
        line: usize,
        record: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("line {line}: field {field} is not a valid number: {value}")]
    InvalidNumber {
        line: usize,
        field: usize,
        value: String,
    },

    #[error("duplicate vertex id {id}")]
    DuplicateVertex { id: usize },

    #[error("line {line}: quaternion has norm {norm:.6}, expected 1")]
    InvalidQuaternion { line: usize, norm: f64 },

    #[error("graph file {} has no extension", path.display())]
    MissingExtension { path: PathBuf },

    #[error("graph file {} has the unsupported extension {extension}", path.display())]
    UnsupportedExtension { path: PathBuf, extension: String },

    #[error("the graph has no {kind} vertices")]
    NoVertices { kind: &'static str },

    #[error(
        "the TORO format holds only SE2 graphs, this one has {vertices} SE3 vertices and {edges} SE3 edges"
    )]
    ToroSe3 { vertices: usize, edges: usize },
}

impl GraphError {
    /// A failed read of the graph file at `path`.
    pub fn read(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Read {
            path: path.into(),
            source,
        }
    }

    /// A failed write of the graph file at `path`.
    pub fn write(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Write {
            path: path.into(),
            source,
        }
    }
}

/// Parse token `field` of a whitespace-split record as `T`.
///
/// The caller checks the record length first, so a missing token here is
/// reported as a short record of `field + 1` fields.
pub(crate) fn parse_field<T: std::str::FromStr>(
    parts: &[&str],
    field: usize,
    line: usize,
    record: &'static str,
) -> GraphResult<T> {
    let token = parts.get(field).ok_or(GraphError::MissingFields {
        line,
        record,
        expected: field + 1,
        actual: parts.len(),
    })?;
    token.parse::<T>().map_err(|_| GraphError::InvalidNumber {
        line,
        field,
        value: (*token).to_string(),
    })
}

/// Fail with [`GraphError::MissingFields`] when `parts` is shorter than `expected`.
pub(crate) fn require_fields(
    parts: &[&str],
    expected: usize,
    line: usize,
    record: &'static str,
) -> GraphResult<()> {
    if parts.len() < expected {
        return Err(GraphError::MissingFields {
            line,
            record,
            expected,
            actual: parts.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_strings_name_their_values() {
        let missing = GraphError::MissingFields {
            line: 4,
            record: "EDGE_SE2",
            expected: 12,
            actual: 7,
        };
        assert_eq!(
            missing.to_string(),
            "line 4: EDGE_SE2 has 7 fields, expected at least 12"
        );
        let number = GraphError::InvalidNumber {
            line: 2,
            field: 3,
            value: "abc".to_string(),
        };
        assert_eq!(
            number.to_string(),
            "line 2: field 3 is not a valid number: abc"
        );
    }

    #[test]
    fn parse_field_reports_the_bad_token() {
        let parts = ["VERTEX_SE2", "1", "x"];
        let result = parse_field::<f64>(&parts, 2, 9, "VERTEX_SE2");
        assert!(matches!(
            result,
            Err(GraphError::InvalidNumber { line: 9, field: 2, ref value }) if value == "x"
        ));
        let missing = parse_field::<f64>(&parts, 5, 9, "VERTEX_SE2");
        assert!(matches!(
            missing,
            Err(GraphError::MissingFields {
                expected: 6,
                actual: 3,
                ..
            })
        ));
    }
}
