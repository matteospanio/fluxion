//! Versioned, self-describing wrapper for fluxion's on-disk artifacts (plan task B9).
//!
//! `.fxg` graphs ([`crate::fxg`]) and frozen plans ([`crate::FrozenSos`]) are serialized inside an
//! [`Envelope`] — `{version, kind, fs, payload}`. As the op set / format evolves an old or wrong-type
//! file is then rejected with a clear error instead of **silently mis-decoding** (`OpKind` is
//! `#[non_exhaustive]`, so it *will* change). `kind` tags the payload; `fs` records the sample rate
//! for coefficient artifacts (a graph is fs-agnostic).

use std::fmt;

use serde::{Deserialize, Serialize};

/// Current on-disk format version. Bump on any breaking change to a serialized payload shape (a
/// removed/renamed `OpKind`, a changed field layout, …); files written by a newer version are then
/// rejected here rather than mis-parsed.
pub const FORMAT_VERSION: u32 = 1;

/// A versioned wrapper around a serialized artifact: `{version, kind, fs, payload}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Envelope<T> {
    /// Format version the file was written with (see [`FORMAT_VERSION`]).
    pub version: u32,
    /// Payload tag, e.g. `"graph"` or `"frozen-sos"` — guards against loading the wrong type.
    pub kind: String,
    /// Sample rate (Hz) for coefficient artifacts; `None` for an fs-agnostic graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fs: Option<u32>,
    /// The wrapped artifact.
    pub payload: T,
}

impl<T> Envelope<T> {
    /// Wrap `payload` at the current [`FORMAT_VERSION`].
    pub fn new(kind: &str, fs: Option<u32>, payload: T) -> Self {
        Self {
            version: FORMAT_VERSION,
            kind: kind.to_string(),
            fs,
            payload,
        }
    }

    /// Validate the version and payload `kind` after deserialization; returns the mismatch, if any.
    pub fn check(&self, expected_kind: &str) -> Result<(), EnvelopeError> {
        if self.version > FORMAT_VERSION {
            return Err(EnvelopeError::Version {
                found: self.version,
                current: FORMAT_VERSION,
            });
        }
        if self.kind != expected_kind {
            return Err(EnvelopeError::Kind {
                found: self.kind.clone(),
                expected: expected_kind.to_string(),
            });
        }
        Ok(())
    }
}

/// A version or payload-kind mismatch when loading a fluxion artifact.
#[derive(Clone, Debug, PartialEq)]
pub enum EnvelopeError {
    /// The file was written by a newer format version than this build understands.
    Version {
        /// Version found in the file.
        found: u32,
        /// Highest version this build understands ([`FORMAT_VERSION`]).
        current: u32,
    },
    /// The file's payload kind is not the one being loaded.
    Kind {
        /// Payload kind found in the file.
        found: String,
        /// Payload kind that was expected.
        expected: String,
    },
}

impl fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnvelopeError::Version { found, current } => write!(
                f,
                "unsupported format version {found} (this build understands up to {current}) — \
                 re-export with a matching fluxion version"
            ),
            EnvelopeError::Kind { found, expected } => {
                write!(
                    f,
                    "expected a '{expected}' artifact but the file is '{found}'"
                )
            }
        }
    }
}

impl std::error::Error for EnvelopeError {}

/// Failure loading a fluxion artifact: malformed JSON (including a pre-envelope, unversioned file,
/// whose missing `version`/`kind` fields surface as a JSON error), or a version/kind mismatch.
#[derive(Debug)]
pub enum LoadError {
    /// The bytes were not valid JSON for the expected envelope shape.
    Json(serde_json::Error),
    /// The envelope decoded but its version or kind is incompatible.
    Envelope(EnvelopeError),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Json(e) => write!(f, "malformed artifact: {e}"),
            LoadError::Envelope(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<serde_json::Error> for LoadError {
    fn from(e: serde_json::Error) -> Self {
        LoadError::Json(e)
    }
}
impl From<EnvelopeError> for LoadError {
    fn from(e: EnvelopeError) -> Self {
        LoadError::Envelope(e)
    }
}

#[cfg(test)]
mod tests {
    use super::{Envelope, EnvelopeError, FORMAT_VERSION};

    #[test]
    fn round_trips_and_omits_none_fs() {
        let env = Envelope::new("graph", None, vec![1u32, 2, 3]);
        let json = serde_json::to_string(&env).unwrap();
        assert!(!json.contains("\"fs\""), "None fs should be omitted");
        let back: Envelope<Vec<u32>> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, env);
        assert!(back.check("graph").is_ok());
    }

    #[test]
    fn check_rejects_future_version_and_wrong_kind() {
        let mut env = Envelope::new("graph", Some(48_000), 7u8);
        assert!(env.check("frozen-sos").is_err()); // wrong kind
        env.version = FORMAT_VERSION + 1;
        assert!(matches!(
            env.check("graph"),
            Err(EnvelopeError::Version { .. })
        ));
    }
}
