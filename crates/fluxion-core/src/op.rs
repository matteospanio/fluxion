//! The typed op catalog ([`OpKind`]) and concrete op instances ([`Op`]).

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::param::{ParamSpec, Unit};

/// The kind of a DSP leaf op. Each kind has a fixed parameter schema ([`OpKind::params`]).
///
/// This is the typed replacement for the earlier string-keyed placeholder: it makes the IR
/// self-describing (names, units, defaults, bounds) for validation, the CLI parser, and lowering.
/// New ops are added here as the `fluxion-ops` crate grows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OpKind {
    /// Linear gain, `y = x * gain`.
    Gain,
    /// Butterworth low-pass filter (`cutoff` Hz, integer `order`). Exposed as the `Lo` variant.
    Lowpass,
    /// Butterworth high-pass filter (`cutoff` Hz, integer `order`). Exposed as the `Hi` variant.
    Highpass,
    /// Peak normalization to a target linear `peak`.
    Normalize,
    /// RBJ peaking EQ: `gain` dB around `frequency` with bandwidth `q`.
    Peaking,
    /// RBJ low shelf: `gain` dB below `frequency` (bandwidth `q`).
    LowShelf,
    /// RBJ high shelf: `gain` dB above `frequency` (bandwidth `q`).
    HighShelf,
    /// RBJ notch at `frequency` with bandwidth `q`.
    Notch,
    /// RBJ band-pass (0 dB peak) at `frequency` with bandwidth `q`.
    Bandpass,
    /// RBJ all-pass at `frequency` with bandwidth `q`.
    Allpass,
    /// Single delayed tap crossfaded with the dry signal (`time` s, `mix`).
    Delay,
    /// Feedback echo: `wet` repeating echoes spaced `time` s apart with `feedback`.
    Echo,
}

// Static parameter tables — one per kind.
static GAIN_PARAMS: [ParamSpec; 1] = [ParamSpec::new(
    "gain",
    Unit::Linear,
    1.0,
    f32::NEG_INFINITY,
    f32::INFINITY,
)];
static LOWPASS_PARAMS: [ParamSpec; 2] = [
    ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
    ParamSpec::new("order", Unit::Linear, 2.0, 1.0, 16.0),
];
static HIGHPASS_PARAMS: [ParamSpec; 2] = [
    ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
    ParamSpec::new("order", Unit::Linear, 2.0, 1.0, 16.0),
];
static NORMALIZE_PARAMS: [ParamSpec; 1] = [ParamSpec::new(
    "peak",
    Unit::Linear,
    1.0,
    0.0,
    f32::INFINITY,
)];
// frequency + gain + q (peaking and both shelves share this schema).
static PEQ_PARAMS: [ParamSpec; 3] = [
    ParamSpec::new("frequency", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
    ParamSpec::new("gain", Unit::Db, 0.0, f32::NEG_INFINITY, f32::INFINITY),
    ParamSpec::new("q", Unit::Q, 0.707, 1e-3, 1000.0),
];
// frequency + q (notch, bandpass, allpass share this schema).
static FQ_PARAMS: [ParamSpec; 2] = [
    ParamSpec::new("frequency", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
    ParamSpec::new("q", Unit::Q, 0.707, 1e-3, 1000.0),
];
static DELAY_PARAMS: [ParamSpec; 2] = [
    ParamSpec::new("time", Unit::Seconds, 0.25, 0.0, 60.0),
    ParamSpec::new("mix", Unit::Linear, 0.5, 0.0, 1.0),
];
static ECHO_PARAMS: [ParamSpec; 3] = [
    ParamSpec::new("time", Unit::Seconds, 0.25, 0.0, 60.0),
    ParamSpec::new("feedback", Unit::Linear, 0.3, 0.0, 0.99),
    ParamSpec::new("wet", Unit::Linear, 0.5, 0.0, 1.0),
];

impl OpKind {
    /// Stable identifier used in the DSL / CLI / `.fxg`, e.g. `"lowpass"`.
    pub fn name(self) -> &'static str {
        match self {
            OpKind::Gain => "gain",
            OpKind::Lowpass => "lowpass",
            OpKind::Highpass => "highpass",
            OpKind::Normalize => "normalize",
            OpKind::Peaking => "peaking",
            OpKind::LowShelf => "lowshelf",
            OpKind::HighShelf => "highshelf",
            OpKind::Notch => "notch",
            OpKind::Bandpass => "bandpass",
            OpKind::Allpass => "allpass",
            OpKind::Delay => "delay",
            OpKind::Echo => "echo",
        }
    }

    /// The parameter schema for this op, in positional order.
    pub fn params(self) -> &'static [ParamSpec] {
        match self {
            OpKind::Gain => &GAIN_PARAMS,
            OpKind::Lowpass => &LOWPASS_PARAMS,
            OpKind::Highpass => &HIGHPASS_PARAMS,
            OpKind::Normalize => &NORMALIZE_PARAMS,
            OpKind::Peaking | OpKind::LowShelf | OpKind::HighShelf => &PEQ_PARAMS,
            OpKind::Notch | OpKind::Bandpass | OpKind::Allpass => &FQ_PARAMS,
            OpKind::Delay => &DELAY_PARAMS,
            OpKind::Echo => &ECHO_PARAMS,
        }
    }

    /// Look up a kind by its DSL name (inverse of [`OpKind::name`]).
    pub fn from_name(name: &str) -> Option<OpKind> {
        OpKind::all().iter().copied().find(|k| k.name() == name)
    }

    /// Every op kind, for enumeration (CLI help, validation).
    pub fn all() -> &'static [OpKind] {
        &[
            OpKind::Gain,
            OpKind::Lowpass,
            OpKind::Highpass,
            OpKind::Normalize,
            OpKind::Peaking,
            OpKind::LowShelf,
            OpKind::HighShelf,
            OpKind::Notch,
            OpKind::Bandpass,
            OpKind::Allpass,
            OpKind::Delay,
            OpKind::Echo,
        ]
    }

    /// The default parameter vector (one entry per [`ParamSpec`]).
    pub fn defaults(self) -> Vec<f32> {
        self.params().iter().map(|p| p.default).collect()
    }
}

/// Error from constructing or validating an [`Op`].
#[derive(Clone, Debug, PartialEq)]
pub enum OpError {
    /// Wrong number of parameters for the kind.
    Arity {
        /// The op whose arity was violated.
        kind: OpKind,
        /// Number of parameters the kind expects.
        expected: usize,
        /// Number of parameters supplied.
        got: usize,
    },
    /// A parameter was NaN or outside its static bounds.
    OutOfRange {
        /// The op whose parameter was invalid.
        kind: OpKind,
        /// Name of the offending parameter.
        param: &'static str,
        /// The supplied value.
        value: f32,
        /// Inclusive lower bound.
        min: f32,
        /// Inclusive upper bound.
        max: f32,
    },
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpError::Arity {
                kind,
                expected,
                got,
            } => write!(
                f,
                "op '{}' expects {expected} parameter(s), got {got}",
                kind.name()
            ),
            OpError::OutOfRange {
                kind,
                param,
                value,
                min,
                max,
            } => write!(
                f,
                "op '{}' parameter '{param}' = {value} is out of range [{min}, {max}]",
                kind.name()
            ),
        }
    }
}

impl std::error::Error for OpError {}

/// A concrete leaf op: an [`OpKind`] plus its positional parameter values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Op {
    /// What this op does.
    pub kind: OpKind,
    /// Parameter values, in the order of [`OpKind::params`].
    pub params: Vec<f32>,
}

impl Op {
    /// Validating constructor: checks arity and that each value is non-NaN and within bounds.
    ///
    /// ```
    /// use fluxion_core::{Op, OpKind};
    /// assert!(Op::new(OpKind::Lowpass, [800.0, 4.0]).is_ok());
    /// assert!(Op::new(OpKind::Lowpass, [800.0]).is_err());  // wrong arity
    /// assert!(Op::new(OpKind::Lowpass, [-1.0, 2.0]).is_err()); // cutoff out of range
    /// ```
    pub fn new(kind: OpKind, params: impl Into<Vec<f32>>) -> Result<Op, OpError> {
        let params = params.into();
        let specs = kind.params();
        if params.len() != specs.len() {
            return Err(OpError::Arity {
                kind,
                expected: specs.len(),
                got: params.len(),
            });
        }
        for (spec, &v) in specs.iter().zip(&params) {
            if v.is_nan() || v < spec.min || v > spec.max {
                return Err(OpError::OutOfRange {
                    kind,
                    param: spec.name,
                    value: v,
                    min: spec.min,
                    max: spec.max,
                });
            }
        }
        Ok(Op { kind, params })
    }
}

#[cfg(test)]
mod tests {
    use super::{Op, OpKind};

    #[test]
    fn name_roundtrips() {
        for &k in OpKind::all() {
            assert_eq!(OpKind::from_name(k.name()), Some(k));
        }
        assert_eq!(OpKind::from_name("nope"), None);
    }

    #[test]
    fn defaults_match_arity() {
        assert_eq!(OpKind::Lowpass.defaults(), vec![1000.0, 2.0]);
        assert_eq!(OpKind::Gain.defaults(), vec![1.0]);
        assert_eq!(OpKind::Peaking.defaults().len(), 3);
        assert_eq!(OpKind::Notch.defaults().len(), 2);
    }

    #[test]
    fn validation_rejects_bad_arity_and_range() {
        assert!(Op::new(OpKind::Gain, [1.0]).is_ok());
        assert!(Op::new(OpKind::Gain, []).is_err());
        assert!(Op::new(OpKind::Lowpass, [-5.0, 2.0]).is_err());
        assert!(Op::new(OpKind::Lowpass, [1000.0, f32::NAN]).is_err());
        assert!(Op::new(OpKind::Peaking, [1000.0, 6.0, 0.0]).is_err()); // q below min
    }
}
