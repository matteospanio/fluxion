//! `.fxg` — save and load a reified [`Graph`] as JSON.
//!
//! This serializes the *graph* (ops + parameters), not frozen coefficients; the realtime
//! freeze/export (plan task G2) builds on top of it later. The on-disk shape is serde's default
//! JSON encoding of [`Graph`] and is **not yet a stable format** — treat it as an internal artifact
//! until v1.0.

use std::io;
use std::path::Path;

use crate::Graph;

/// Serialize a graph to a pretty JSON string.
pub fn to_json(graph: &Graph) -> String {
    // A DSP graph contains only enums, structs, and `Vec<f32>` — serialization cannot fail.
    serde_json::to_string_pretty(graph).expect("graph is always serializable")
}

/// Parse a graph from a JSON string.
pub fn from_json(s: &str) -> serde_json::Result<Graph> {
    serde_json::from_str(s)
}

/// Write a graph to a `.fxg` file.
pub fn save(graph: &Graph, path: impl AsRef<Path>) -> io::Result<()> {
    std::fs::write(path, to_json(graph))
}

/// Read a graph from a `.fxg` file.
pub fn load(path: impl AsRef<Path>) -> io::Result<Graph> {
    let s = std::fs::read_to_string(path)?;
    from_json(&s).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::{from_json, to_json};
    use crate::{Graph, OpKind};

    #[test]
    fn json_roundtrip_preserves_graph() {
        let g = (Graph::op(OpKind::Lowpass, [800.0, 4.0])
            + Graph::op(OpKind::Highpass, [80.0, 2.0]))
            | Graph::op(OpKind::Peaking, [1_000.0, 6.0, 1.5]);
        assert_eq!(from_json(&to_json(&g)).unwrap(), g);
    }
}
