//! Frozen realtime plan (plan task G2).
//!
//! Where [`fxg`](crate::fxg) serializes a [`Graph`](crate::Graph) — the design *parameters* (cutoff,
//! order, …) — a [`FrozenSos`] stores the *designed coefficients* at a fixed sample rate, so a
//! realtime executor can run it with no filter design at load time. The design step (graph →
//! `FrozenSos`) lives in `fluxion-backend::freeze`; running it is `fluxion-rt::SosStream`.

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::envelope::{Envelope, LoadError};

/// The envelope `kind` tag for a serialized frozen plan.
const FROZEN_KIND: &str = "frozen-sos";

/// A frozen SOS cascade: the designed sections plus the sample rate they were designed for. Each
/// section is `[b0, b1, b2, a1, a2]` (the denominator `a0` is normalized to 1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrozenSos {
    /// Sample rate (Hz) the coefficients were designed for.
    pub fs: u32,
    /// Second-order sections, each `[b0, b1, b2, a1, a2]`.
    pub sections: Vec<[f32; 5]>,
}

impl FrozenSos {
    /// Build a frozen plan from sections at sample rate `fs`.
    pub fn new(fs: u32, sections: Vec<[f32; 5]>) -> Self {
        Self { fs, sections }
    }

    /// Serialize to pretty JSON, wrapped in a versioned [`Envelope`] (with `fs`; cannot fail).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&Envelope::new(FROZEN_KIND, Some(self.fs), self))
            .expect("frozen plan is always serializable")
    }

    /// Parse from JSON, validating the envelope version and kind.
    pub fn from_json(s: &str) -> Result<Self, LoadError> {
        let env: Envelope<FrozenSos> = serde_json::from_str(s)?;
        env.check(FROZEN_KIND)?;
        Ok(env.payload)
    }

    /// Write to a `.fxg`-style file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    /// Read from a file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_json(&s).map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::FrozenSos;

    #[test]
    fn json_round_trip() {
        let plan = FrozenSos::new(
            48_000,
            vec![[1.0, 0.5, 0.25, -0.3, 0.1], [0.8, 0.0, 0.0, -0.2, 0.0]],
        );
        let back = FrozenSos::from_json(&plan.to_json()).unwrap();
        assert_eq!(plan, back);
    }

    #[test]
    fn envelope_carries_fs_and_rejects_wrong_kind() {
        let plan = FrozenSos::new(44_100, vec![[1.0, 0.0, 0.0, -0.2, 0.0]]);
        let json = plan.to_json();
        assert!(json.contains("\"frozen-sos\"") && json.contains("\"fs\": 44100"));
        // A graph envelope loaded as a frozen plan is rejected, not mis-decoded.
        let wrong = json.replace("\"frozen-sos\"", "\"graph\"");
        assert!(FrozenSos::from_json(&wrong).is_err());
    }
}
