//! `.fxg` — save and load a reified [`Graph`] as JSON.
//!
//! This serializes the *graph* (ops + parameters), not frozen coefficients; the realtime
//! freeze/export (plan task G2) builds on top of it later. The graph is wrapped in a versioned
//! [`Envelope`](crate::envelope) (plan task B9), so an old or wrong-type file is rejected with a
//! clear error instead of silently mis-decoding as the op set grows.

use std::io;
use std::path::Path;

use crate::Graph;
use crate::envelope::{Envelope, LoadError};

/// The envelope `kind` tag for a serialized graph.
const GRAPH_KIND: &str = "graph";

/// Serialize a graph to a pretty JSON string (wrapped in a versioned [`Envelope`]).
pub fn to_json(graph: &Graph) -> String {
    // A DSP graph contains only enums, structs, and `Vec<f32>` — serialization cannot fail.
    serde_json::to_string_pretty(&Envelope::new(GRAPH_KIND, None, graph))
        .expect("graph is always serializable")
}

/// Parse a graph from a JSON string, validating the envelope version and kind.
pub fn from_json(s: &str) -> Result<Graph, LoadError> {
    let env: Envelope<Graph> = serde_json::from_str(s)?;
    env.check(GRAPH_KIND)?;
    Ok(env.payload)
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
        let json = to_json(&g);
        assert!(json.contains("\"version\"") && json.contains("\"graph\""));
        assert_eq!(from_json(&json).unwrap(), g);
    }

    #[test]
    fn json_roundtrip_preserves_new_effect_ops() {
        // The compand / overdrive / biquad / reverse / chorus family (incl. the zero-param Reverse)
        // survive an envelope round-trip like every other op.
        let g = Graph::op(OpKind::Compand, [0.01, 0.1, -20.0, 4.0, 6.0, 0.0])
            | Graph::op(OpKind::Overdrive, [20.0, 0.2])
            | Graph::op(OpKind::Biquad, [0.5, -0.2, 0.1, -0.3, 0.05])
            | Graph::op(OpKind::Reverse, [])
            | (Graph::op(OpKind::Chorus, [1.5, 0.002, 0.025, 0.5])
                + Graph::op(OpKind::Flanger, [0.5, 0.002, 0.001, 0.5, 0.5]));
        assert_eq!(from_json(&to_json(&g)).unwrap(), g);
    }

    #[test]
    fn rejects_future_version_wrong_kind_and_pre_envelope() {
        use crate::envelope::{EnvelopeError, LoadError};

        // A future format version is rejected, not mis-decoded.
        let g = Graph::op(OpKind::Gain, [0.5]);
        let bumped = to_json(&g).replace("\"version\": 1", "\"version\": 999");
        assert!(matches!(
            from_json(&bumped),
            Err(LoadError::Envelope(EnvelopeError::Version { .. }))
        ));
        // A frozen-sos envelope loaded as a graph is rejected on kind.
        let wrong = to_json(&g).replace("\"graph\"", "\"frozen-sos\"");
        assert!(matches!(
            from_json(&wrong),
            Err(LoadError::Envelope(EnvelopeError::Kind { .. }))
        ));
        // A pre-envelope (bare-Graph) file fails loudly (missing envelope fields) rather than silently.
        let bare = serde_json::to_string(&g).unwrap();
        assert!(matches!(from_json(&bare), Err(LoadError::Json(_))));
    }
}
